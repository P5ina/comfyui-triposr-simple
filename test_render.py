#!/usr/bin/env python3
"""
Quick test script for TripoSR mesh generation and rendering.
Run on the server: python test_render.py
"""

import os
import sys

# Set up headless rendering BEFORE any OpenGL imports
os.environ["PYOPENGL_PLATFORM"] = "egl"

sys.path.insert(0, 'TripoSR')

import numpy as np
import torch
from PIL import Image
import trimesh

# Patch rembg
import types
rembg = types.ModuleType('rembg')
rembg.remove = lambda x, **kw: x
rembg.new_session = lambda *a, **kw: None
sys.modules['rembg'] = rembg

# Check torchmcubes
try:
    import torchmcubes
    print("[OK] torchmcubes available")
except ImportError:
    print("[ERROR] torchmcubes not installed!")
    sys.exit(1)

import pyrender
import open3d as o3d
from tsr.system import TSR


def resize_foreground(image, ratio=0.85):
    """Resize foreground using alpha channel."""
    image_np = np.array(image)
    if image_np.shape[-1] != 4:
        return image

    alpha = np.where(image_np[..., 3] > 0)
    if len(alpha[0]) == 0:
        return image

    y1, y2 = alpha[0].min(), alpha[0].max()
    x1, x2 = alpha[1].min(), alpha[1].max()

    fg = image_np[y1:y2, x1:x2]
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = np.pad(fg, ((ph0, ph1), (pw0, pw1), (0, 0)), mode="constant", constant_values=0)

    new_size = int(new_image.shape[0] / ratio)
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = np.pad(new_image, ((ph0, ph1), (pw0, pw1), (0, 0)), mode="constant", constant_values=0)

    return Image.fromarray(new_image)


def rgba_to_rgb_gray_bg(image):
    """Convert RGBA to RGB with gray background."""
    image_np = np.array(image).astype(np.float32) / 255.0
    if image_np.shape[-1] != 4:
        return image.convert('RGB')

    rgb = image_np[:, :, :3]
    alpha = image_np[:, :, 3:4]
    composited = rgb * alpha + (1 - alpha) * 0.5
    return Image.fromarray((composited * 255).astype(np.uint8), 'RGB')


def create_camera_pose(azimuth_deg, elevation_deg, distance):
    """Create camera pose matrix."""
    azimuth = np.radians(azimuth_deg)
    elevation = np.radians(elevation_deg)

    x = distance * np.cos(elevation) * np.sin(azimuth)
    y = distance * np.sin(elevation)
    z = distance * np.cos(elevation) * np.cos(azimuth)

    camera_pos = np.array([x, y, z])
    target = np.array([0, 0, 0])
    up = np.array([0, 1, 0])

    forward = target - camera_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = camera_pos

    return pose


def render_mesh_o3d(mesh, texture_image, uvs, azimuth, elevation=20.0, distance=1.5, size=512):
    """Render textured mesh using Open3D."""
    # Create Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()

    # Set up texture
    texture_array = np.array(texture_image)
    # Open3D expects RGB, not RGBA
    if texture_array.shape[2] == 4:
        texture_array = texture_array[:, :, :3]

    # Flip texture vertically for Open3D
    texture_array = np.flipud(texture_array)

    o3d_mesh.textures = [o3d.geometry.Image(texture_array)]
    o3d_mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs[mesh.faces.flatten()])
    o3d_mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(mesh.faces))

    # Set up camera
    azimuth_rad = np.radians(azimuth)
    elevation_rad = np.radians(elevation)

    x = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
    y = distance * np.sin(elevation_rad)
    z = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)

    camera_pos = np.array([x, y, z])
    target = np.array([0, 0, 0])
    up = np.array([0, 1, 0])

    # Create renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(size, size)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

    # Add mesh with material
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLitSSR"  # or "defaultUnlit" for no lighting
    mat.albedo_img = o3d.geometry.Image(texture_array)

    renderer.scene.add_geometry("mesh", o3d_mesh, mat)

    # Set up camera
    renderer.setup_camera(60.0, [0, 0, 0], camera_pos, up)

    # Add lighting
    renderer.scene.scene.set_sun_light([0.5, -1, 0.5], [1, 1, 1], 50000)
    renderer.scene.scene.enable_sun_light(True)

    # Render
    img = renderer.render_to_image()

    return np.asarray(img)


def render_mesh(mesh, azimuth, elevation=20.0, distance=1.5, size=512):
    """Render mesh from given angle using pyrender (vertex colors only)."""
    scene = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.6, 0.6, 0.6])

    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(pyrender_mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_pose = create_camera_pose(azimuth, elevation, distance)
    scene.add(camera, pose=camera_pose)

    key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    scene.add(key_light, pose=create_camera_pose(azimuth - 30, elevation + 30, 1.0))

    fill_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.4)
    scene.add(fill_light, pose=create_camera_pose(azimuth + 60, elevation, 1.0))

    renderer = pyrender.OffscreenRenderer(size, size)
    try:
        color, _ = renderer.render(scene)
    finally:
        renderer.delete()

    return color


def main():
    print("=" * 60)
    print("TripoSR Test Script")
    print("=" * 60)

    # Load model
    print("\n[1/4] Loading TripoSR model...")
    model = TSR.from_pretrained(
        "/workspace/ComfyUI/models/triposr",
        config_name="config.yaml",
        weight_name="model.ckpt"
    )
    model.to("cuda:0")
    model.renderer.set_chunk_size(8192)
    print("      Model loaded!")

    # Load and preprocess image
    print("\n[2/4] Loading and preprocessing image...")
    image_path = "TripoSR/examples/chair.png"
    image = Image.open(image_path)
    print(f"      Original: {image.size}, mode={image.mode}")

    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    image = resize_foreground(image, ratio=0.85)
    image = rgba_to_rgb_gray_bg(image)
    print(f"      Preprocessed: {image.size}, mode={image.mode}")

    # Generate mesh
    print("\n[3/4] Generating 3D mesh...")
    resolution = 256
    texture_resolution = 1024

    with torch.no_grad():
        scene_codes = model([image], device="cuda:0")

        # Try texture baking
        try:
            import moderngl
            from scipy import ndimage

            _orig = moderngl.create_context
            def _egl(*a, **kw):
                kw.setdefault('backend', 'egl')
                return _orig(*a, **kw)
            moderngl.create_context = _egl

            from tsr.bake_texture import bake_texture

            print("      Extracting mesh...")
            meshes = model.extract_mesh(scene_codes, has_vertex_color=False, resolution=resolution)
            mesh = meshes[0]

            print(f"      Baking texture at {texture_resolution}x{texture_resolution}...")
            bake_output = bake_texture(mesh, model, scene_codes[0], texture_resolution)

            new_vertices = mesh.vertices[bake_output["vmapping"]]
            new_faces = bake_output["indices"]
            uvs = bake_output["uvs"]

            # Color correction
            texture_colors = bake_output["colors"].copy()
            rgb = texture_colors[:, :, :3]
            alpha = texture_colors[:, :, 3:4]
            mask = alpha[:, :, 0] > 0.01

            if mask.any():
                rgb_masked = rgb[mask]
                for c in range(3):
                    channel = rgb_masked[:, c]
                    c_min, c_max = np.percentile(channel, [1, 99])
                    if c_max > c_min:
                        rgb[:, :, c] = np.clip((rgb[:, :, c] - c_min) / (c_max - c_min), 0, 1)

                gray = np.mean(rgb, axis=2, keepdims=True)
                rgb = gray + (rgb - gray) * 1.3
                rgb = np.clip(rgb, 0, 1)

                rgb = (rgb - 0.5) * 1.2 + 0.5
                rgb = np.clip(rgb, 0, 1)

            # Inpaint texture to fill background - prevents edge artifacts
            texture_colors = np.concatenate([rgb, alpha], axis=2)
            texture_colors = (texture_colors * 255.0).astype(np.uint8)

            try:
                import cv2
                print("      Inpainting texture background...")
                inpaint_mask = (alpha[:, :, 0] < 0.01).astype(np.uint8) * 255
                rgb_uint8 = texture_colors[:, :, :3]
                inpainted = cv2.inpaint(rgb_uint8, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
                texture_colors[:, :, :3] = inpainted
            except ImportError:
                print("      cv2 not available, skipping inpainting")

            texture_image = Image.fromarray(texture_colors).transpose(Image.FLIP_TOP_BOTTOM)

            # Save texture for inspection
            texture_image.save("/tmp/test_texture.png")
            print("      Saved texture to /tmp/test_texture.png")

            # Create mesh (will render with Open3D using texture)
            mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
            # Store texture info for rendering
            stored_texture = texture_image
            stored_uvs = uvs

            print(f"      Mesh with texture: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

            has_texture = True

        except Exception as e:
            print(f"      Texture baking failed: {e}")
            print("      Falling back to vertex colors...")
            meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=resolution)
            mesh = meshes[0]
            stored_texture = None
            stored_uvs = None
            has_texture = False

    # Transform mesh
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    mesh.vertices -= center

    rot_x = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0], point=[0, 0, 0])
    rot_y = trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0], point=[0, 0, 0])
    mesh.apply_transform(rot_x)
    mesh.apply_transform(rot_y)

    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    scale = 1.0 / np.max(extents) * 1.2
    mesh.vertices *= scale

    # Export mesh
    mesh.export("/tmp/test_mesh.glb")
    print("      Saved mesh to /tmp/test_mesh.glb")

    # Render
    print("\n[4/4] Rendering 8 directions...")
    os.makedirs("/tmp/test_renders", exist_ok=True)
    directions = [("N", 0), ("NE", 45), ("E", 90), ("SE", 135), ("S", 180), ("SW", 225), ("W", 270), ("NW", 315)]

    for name, azimuth in directions:
        print(f"      Rendering {name} ({azimuth}°)...")
        if has_texture and stored_texture is not None:
            color = render_mesh_o3d(mesh, stored_texture, stored_uvs, azimuth, elevation=20.0, distance=1.5, size=512)
        else:
            color = render_mesh(mesh, azimuth, elevation=20.0, distance=1.5, size=512)
        Image.fromarray(color).save(f"/tmp/test_renders/{name}.png")

    print("\n" + "=" * 60)
    print("Done! Output files:")
    print("  - /tmp/test_texture.png (baked texture)")
    print("  - /tmp/test_mesh.glb (3D mesh)")
    print("  - /tmp/test_renders/*.png (rendered views)")
    print("=" * 60)


if __name__ == "__main__":
    main()
