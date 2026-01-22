"""
Common utilities and nodes for 3D sprite generation.

Provides:
- Image preprocessing functions
- Mesh rendering (8 directions)
- Image pixelation
- Sprite sheet combination
"""

import os
import numpy as np
import torch
from PIL import Image
from typing import Tuple, Optional
import trimesh

# Try to import nvdiffrast for GPU texture rendering
try:
    import nvdiffrast.torch as dr
    HAS_NVDIFFRAST = True
except ImportError:
    HAS_NVDIFFRAST = False
    print("[3DSprite] nvdiffrast not available, falling back to pyrender")

# Set headless rendering platform before importing pyrender
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

import pyrender


def bake_texture_to_vertex_colors(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Bake UV texture to vertex colors for rendering compatibility.
    Samples the texture at each vertex's UV coordinate.
    """
    try:
        visual = mesh.visual

        # Check if it's a TextureVisuals with UV mapping
        if hasattr(visual, 'uv') and hasattr(visual, 'material'):
            uv = visual.uv
            material = visual.material

            # Get the texture image
            texture_image = None
            if hasattr(material, 'image') and material.image is not None:
                texture_image = np.array(material.image)
            elif hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                texture_image = np.array(material.baseColorTexture)

            if texture_image is not None and uv is not None:
                h, w = texture_image.shape[:2]

                # Sample texture at UV coordinates
                u = np.clip(uv[:, 0], 0, 1)
                v = np.clip(1 - uv[:, 1], 0, 1)  # Flip V

                px = (u * (w - 1)).astype(int)
                py = (v * (h - 1)).astype(int)

                # Sample colors
                if len(texture_image.shape) == 3:
                    if texture_image.shape[2] == 4:
                        vertex_colors = texture_image[py, px, :]  # RGBA
                    else:
                        rgb = texture_image[py, px, :3]
                        alpha = np.full((len(rgb), 1), 255, dtype=np.uint8)
                        vertex_colors = np.hstack([rgb, alpha])
                else:
                    # Grayscale
                    gray = texture_image[py, px]
                    vertex_colors = np.stack([gray, gray, gray, np.full_like(gray, 255)], axis=-1)

                print(f"[3DSprite] Baked texture to {len(vertex_colors)} vertex colors")

                return trimesh.Trimesh(
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    vertex_colors=vertex_colors
                )

        # Try trimesh's built-in conversion
        if hasattr(visual, 'to_color'):
            colored = visual.to_color()
            if hasattr(colored, 'vertex_colors') and colored.vertex_colors is not None:
                print(f"[3DSprite] Converted visual to vertex colors")
                return trimesh.Trimesh(
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    vertex_colors=colored.vertex_colors
                )

        # Check for existing vertex colors
        if hasattr(visual, 'vertex_colors') and visual.vertex_colors is not None:
            return mesh

    except Exception as e:
        print(f"[3DSprite] Could not bake texture: {e}")

    # Fallback: gray mesh
    print(f"[3DSprite] Warning: No colors found, using gray")
    gray = np.full((len(mesh.vertices), 4), [180, 180, 180, 255], dtype=np.uint8)
    return trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        vertex_colors=gray
    )


class NVDiffrastRenderer:
    """GPU-accelerated renderer using nvdiffrast for proper texture support."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.glctx = dr.RasterizeCudaContext()

    def _make_projection_matrix(self, fov_y: float, aspect: float, near: float, far: float) -> torch.Tensor:
        """Create perspective projection matrix."""
        f = 1.0 / np.tan(fov_y / 2)
        return torch.tensor([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=torch.float32, device=self.device)

    def _make_view_matrix(self, azimuth: float, elevation: float, distance: float) -> torch.Tensor:
        """Create view matrix from spherical coordinates."""
        azimuth_rad = np.radians(azimuth)
        elevation_rad = np.radians(elevation)

        # Camera position
        x = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        y = distance * np.sin(elevation_rad)
        z = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)

        eye = np.array([x, y, z])
        target = np.array([0, 0, 0])
        up = np.array([0, 1, 0])

        # Look-at matrix
        forward = target - eye
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        view = np.eye(4)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[:3, 3] = -np.dot(view[:3, :3], eye)

        return torch.tensor(view, dtype=torch.float32, device=self.device)

    def render(
        self,
        mesh: trimesh.Trimesh,
        azimuth: float,
        elevation: float,
        distance: float,
        size: int,
        bg_color: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """Render mesh with textures using nvdiffrast."""

        # Get mesh data
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=self.device)
        faces = torch.tensor(mesh.faces, dtype=torch.int32, device=self.device)

        # Get texture and UVs if available
        has_texture = False
        texture = None
        uvs = None

        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            uvs = torch.tensor(mesh.visual.uv, dtype=torch.float32, device=self.device)

            if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
                mat = mesh.visual.material
                tex_img = None
                if hasattr(mat, 'image') and mat.image is not None:
                    tex_img = np.array(mat.image)
                elif hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                    tex_img = np.array(mat.baseColorTexture)

                if tex_img is not None:
                    # Ensure RGBA
                    if len(tex_img.shape) == 2:
                        tex_img = np.stack([tex_img] * 3 + [np.full_like(tex_img, 255)], axis=-1)
                    elif tex_img.shape[2] == 3:
                        tex_img = np.concatenate([tex_img, np.full(tex_img.shape[:2] + (1,), 255, dtype=np.uint8)], axis=-1)

                    texture = torch.tensor(tex_img, dtype=torch.float32, device=self.device) / 255.0
                    has_texture = True

        # Get vertex colors as fallback
        vertex_colors = None
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            vc = mesh.visual.vertex_colors
            if len(vc.shape) == 2 and vc.shape[1] >= 3:
                vertex_colors = torch.tensor(vc[:, :4] if vc.shape[1] >= 4 else np.hstack([vc[:, :3], np.full((len(vc), 1), 255)]),
                                           dtype=torch.float32, device=self.device) / 255.0

        # Build MVP matrix
        proj = self._make_projection_matrix(np.pi / 3, 1.0, 0.1, 100.0)
        view = self._make_view_matrix(azimuth, elevation, distance)
        mvp = proj @ view

        # Transform vertices to clip space
        vertices_h = torch.cat([vertices, torch.ones(vertices.shape[0], 1, device=self.device)], dim=1)
        vertices_clip = (mvp @ vertices_h.T).T

        # Rasterize
        vertices_clip = vertices_clip.unsqueeze(0).contiguous()
        faces = faces.unsqueeze(0).contiguous()

        rast, _ = dr.rasterize(self.glctx, vertices_clip, faces, resolution=[size, size])

        # Interpolate colors/textures
        if has_texture and uvs is not None:
            # Interpolate UVs
            uvs_h = uvs.unsqueeze(0).contiguous()
            uv_interp, _ = dr.interpolate(uvs_h, rast, faces)

            # Sample texture
            uv_for_sample = uv_interp[0, :, :, :2]  # (H, W, 2)
            uv_for_sample = uv_for_sample * 2 - 1  # Convert to [-1, 1]
            uv_for_sample = uv_for_sample.unsqueeze(0)  # (1, H, W, 2)

            texture_4d = texture.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            color = torch.nn.functional.grid_sample(texture_4d, uv_for_sample, mode='bilinear', padding_mode='border', align_corners=True)
            color = color.squeeze(0).permute(1, 2, 0)  # (H, W, C)

        elif vertex_colors is not None:
            # Interpolate vertex colors
            vc_h = vertex_colors.unsqueeze(0).contiguous()
            color, _ = dr.interpolate(vc_h, rast, faces)
            color = color[0]  # (H, W, C)

        else:
            # Gray fallback
            color = torch.full((size, size, 4), 0.7, device=self.device)

        # Apply background where no geometry
        mask = rast[0, :, :, 3:4] > 0
        bg = torch.tensor(bg_color, device=self.device).view(1, 1, 4)
        color = torch.where(mask, color, bg.expand(size, size, 4))

        # Convert to numpy uint8
        color = (color.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

        # RGB only (drop alpha for output)
        return color[:, :, :3]


# Global renderer instance
_nvdiffrast_renderer: Optional[NVDiffrastRenderer] = None


def get_nvdiffrast_renderer() -> Optional[NVDiffrastRenderer]:
    """Get or create global nvdiffrast renderer."""
    global _nvdiffrast_renderer
    if not HAS_NVDIFFRAST:
        return None
    if _nvdiffrast_renderer is None:
        try:
            _nvdiffrast_renderer = NVDiffrastRenderer()
            print("[3DSprite] Initialized nvdiffrast renderer")
        except Exception as e:
            print(f"[3DSprite] Could not initialize nvdiffrast: {e}")
            return None
    return _nvdiffrast_renderer


def resize_foreground(image: Image.Image, ratio: float = 0.85) -> Image.Image:
    """
    Resize foreground using alpha channel to find object bounds.

    Args:
        image: PIL Image in RGBA mode with transparent background
        ratio: Target occupancy ratio (0.85 = object fills 85% of the image)

    Returns:
        RGBA image with foreground centered and padded
    """
    image_np = np.array(image)

    if image_np.shape[-1] != 4:
        print("[3DSprite] Warning: resize_foreground expects RGBA image, got", image_np.shape)
        return image

    # Find non-transparent pixels using alpha channel
    alpha = np.where(image_np[..., 3] > 0)

    if len(alpha[0]) == 0:
        print("[3DSprite] Warning: No foreground pixels found (all transparent)")
        return image

    # Extract bounding box of foreground
    y1, y2 = alpha[0].min(), alpha[0].max()
    x1, x2 = alpha[1].min(), alpha[1].max()

    # Crop to foreground
    fg = image_np[y1:y2, x1:x2]

    # Pad to square
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = np.pad(
        fg,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    # Add padding according to ratio
    new_size = int(new_image.shape[0] / ratio)
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = np.pad(
        new_image,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    return Image.fromarray(new_image)


def rgba_to_rgb_white_background(image: Image.Image) -> Image.Image:
    """
    Convert RGBA to RGB by compositing with white background.

    Args:
        image: PIL Image in RGBA mode

    Returns:
        RGB image with white background where transparency was
    """
    image_np = np.array(image).astype(np.float32) / 255.0

    if image_np.shape[-1] != 4:
        return image.convert('RGB')

    rgb = image_np[:, :, :3]
    alpha = image_np[:, :, 3:4]

    # Composite with white (1.0) background - required by Hunyuan3D
    composited = rgb * alpha + (1 - alpha) * 1.0

    result = (composited * 255.0).astype(np.uint8)
    return Image.fromarray(result, 'RGB')


# Keep alias for backwards compatibility
rgba_to_rgb_gray_background = rgba_to_rgb_white_background


class RenderMesh8Directions:
    """
    Renders a mesh from 8 cardinal and intercardinal directions.
    Returns 8 separate images: N, NE, E, SE, S, SW, W, NW.
    """

    CATEGORY = "3DSprite"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    FUNCTION = "render"

    DIRECTIONS = [
        ("N", 0),
        ("NE", 45),
        ("E", 90),
        ("SE", 135),
        ("S", 180),
        ("SW", 225),
        ("W", 270),
        ("NW", 315),
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "render_size": ("INT", {"default": 512, "min": 64, "max": 1024, "step": 64}),
                "elevation": ("FLOAT", {"default": 20.0, "min": -90.0, "max": 90.0, "step": 5.0}),
                "distance": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 10.0, "step": 0.1}),
                "background_color": (["white", "black", "transparent"],),
                "auto_align": ("BOOLEAN", {"default": False}),
                "mesh_pitch": ("FLOAT", {"default": 0.0, "min": -45.0, "max": 45.0, "step": 1.0}),
                "mesh_roll": ("FLOAT", {"default": 0.0, "min": -45.0, "max": 45.0, "step": 1.0}),
            }
        }

    def _create_camera_pose(self, azimuth_deg: float, elevation_deg: float, distance: float) -> np.ndarray:
        """Create camera pose matrix for given azimuth and elevation angles."""
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

    def _render_single_view(
        self,
        mesh: trimesh.Trimesh,
        azimuth: float,
        elevation: float,
        distance: float,
        size: int,
        bg_color: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """Render mesh from a single viewpoint."""

        # Try nvdiffrast first (better texture support)
        nvrenderer = get_nvdiffrast_renderer()
        if nvrenderer is not None:
            try:
                return nvrenderer.render(mesh, azimuth, elevation, distance, size, bg_color)
            except Exception as e:
                print(f"[3DSprite] nvdiffrast render failed: {e}, falling back to pyrender")

        # Fallback to pyrender with baked vertex colors
        render_mesh = bake_texture_to_vertex_colors(mesh)

        scene = pyrender.Scene(bg_color=bg_color, ambient_light=[0.6, 0.6, 0.6])

        pyrender_mesh = pyrender.Mesh.from_trimesh(render_mesh, smooth=False)
        scene.add(pyrender_mesh)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        camera_pose = self._create_camera_pose(azimuth, elevation, distance)
        scene.add(camera, pose=camera_pose)

        # Key light
        key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
        key_pose = self._create_camera_pose(azimuth - 30, elevation + 30, 1.0)
        scene.add(key_light, pose=key_pose)

        # Fill light
        fill_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.4)
        fill_pose = self._create_camera_pose(azimuth + 60, elevation, 1.0)
        scene.add(fill_light, pose=fill_pose)

        renderer = pyrender.OffscreenRenderer(size, size)
        try:
            color, _ = renderer.render(scene)
        finally:
            renderer.delete()

        return color

    def render(
        self,
        mesh: trimesh.Trimesh,
        render_size: int,
        elevation: float,
        distance: float,
        background_color: str,
        auto_align: bool,
        mesh_pitch: float,
        mesh_roll: float
    ):
        bg_colors = {
            "white": (1.0, 1.0, 1.0, 1.0),
            "black": (0.0, 0.0, 0.0, 1.0),
            "transparent": (0.0, 0.0, 0.0, 0.0),
        }
        bg_color = bg_colors.get(background_color, (1.0, 1.0, 1.0, 1.0))

        # Center the mesh
        mesh_centered = mesh.copy()
        bounds = mesh_centered.bounds
        center = (bounds[0] + bounds[1]) / 2
        mesh_centered.vertices -= center

        if auto_align:
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=3)
                pca.fit(mesh_centered.vertices)

                components = pca.components_
                variances = pca.explained_variance_
                up_idx = np.argmin(variances)

                axes = list(range(3))
                axes.remove(up_idx)

                rotation = np.eye(4)
                rotation[:3, 1] = components[up_idx]
                rotation[:3, 0] = components[axes[0]]
                rotation[:3, 2] = components[axes[1]]

                if np.linalg.det(rotation[:3, :3]) < 0:
                    rotation[:3, 2] *= -1

                mesh_centered.apply_transform(rotation.T)

                rot_180 = trimesh.transformations.rotation_matrix(
                    np.pi, [0, 1, 0], point=[0, 0, 0]
                )
                mesh_centered.apply_transform(rot_180)
                print("[3DSprite] Auto-aligned mesh using PCA")
            except Exception as e:
                print(f"[3DSprite] Auto-align failed: {e}")
                rot_x = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0], point=[0, 0, 0])
                rot_y = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0], point=[0, 0, 0])
                mesh_centered.apply_transform(rot_x)
                mesh_centered.apply_transform(rot_y)
        else:
            rot_x = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0], point=[0, 0, 0])
            rot_y = trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0], point=[0, 0, 0])
            mesh_centered.apply_transform(rot_x)
            mesh_centered.apply_transform(rot_y)

        # Apply user corrections
        if mesh_pitch != 0.0:
            rot_pitch = trimesh.transformations.rotation_matrix(
                np.radians(mesh_pitch), [1, 0, 0], point=[0, 0, 0]
            )
            mesh_centered.apply_transform(rot_pitch)
        if mesh_roll != 0.0:
            rot_roll = trimesh.transformations.rotation_matrix(
                np.radians(mesh_roll), [0, 0, 1], point=[0, 0, 0]
            )
            mesh_centered.apply_transform(rot_roll)

        # Scale to fit
        bounds = mesh_centered.bounds
        extents = bounds[1] - bounds[0]
        max_extent = np.max(extents)
        scale = 1.0 / max_extent
        mesh_centered.vertices *= scale * 1.2

        rendered_images = []

        for direction_name, azimuth in self.DIRECTIONS:
            print(f"[3DSprite] Rendering {direction_name} ({azimuth}deg)...")

            color = self._render_single_view(
                mesh_centered,
                azimuth,
                elevation,
                distance,
                render_size,
                bg_color
            )

            img_tensor = torch.from_numpy(color.astype(np.float32) / 255.0)
            img_tensor = img_tensor.unsqueeze(0)

            rendered_images.append(img_tensor)

        print(f"[3DSprite] All 8 directions rendered at {render_size}x{render_size}")

        return tuple(rendered_images)


class PixelateImage:
    """
    Pixelates an image with optional color quantization.
    """

    CATEGORY = "3DSprite"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "pixelate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pixel_resolution": ([32, 64, 128, 256],),
                "color_count": ([8, 16, 32, 64, 128, 256],),
                "output_size": ("INT", {"default": 512, "min": 32, "max": 1024, "step": 32}),
            }
        }

    def _quantize_colors_kmeans(self, img: Image.Image, n_colors: int) -> Image.Image:
        """Quantize colors using K-means clustering."""
        img_array = np.array(img)
        original_shape = img_array.shape

        has_alpha = img_array.shape[2] == 4 if len(img_array.shape) > 2 else False

        if has_alpha:
            rgb = img_array[:, :, :3]
            alpha = img_array[:, :, 3]
        else:
            rgb = img_array
            alpha = None

        pixels = rgb.reshape(-1, 3).astype(np.float32)

        try:
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, n_init=3)
            labels = kmeans.fit_predict(pixels)
            centers = kmeans.cluster_centers_
        except ImportError:
            print("[3DSprite] sklearn not available, using PIL quantize")
            if has_alpha:
                img_rgb = Image.fromarray(rgb)
            else:
                img_rgb = img
            quantized = img_rgb.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)
            quantized = quantized.convert('RGB')
            if has_alpha:
                quantized = quantized.convert('RGBA')
                quantized.putalpha(Image.fromarray(alpha))
            return quantized

        quantized_pixels = centers[labels].astype(np.uint8)
        quantized_rgb = quantized_pixels.reshape(original_shape[:2] + (3,))

        if has_alpha:
            quantized_array = np.dstack([quantized_rgb, alpha])
            return Image.fromarray(quantized_array, 'RGBA')
        else:
            return Image.fromarray(quantized_rgb, 'RGB')

    def pixelate(
        self,
        image: torch.Tensor,
        pixel_resolution: int,
        color_count: int,
        output_size: int
    ):
        img_np = image[0].cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        if img_np.shape[2] == 4:
            pil_image = Image.fromarray(img_np, 'RGBA')
        else:
            pil_image = Image.fromarray(img_np, 'RGB')

        original_size = pil_image.size

        small = pil_image.resize(
            (pixel_resolution, pixel_resolution),
            Image.Resampling.NEAREST
        )

        quantized = self._quantize_colors_kmeans(small, color_count)

        result = quantized.resize(
            (output_size, output_size),
            Image.Resampling.NEAREST
        )

        result_np = np.array(result).astype(np.float32) / 255.0

        if result_np.shape[2] == 4:
            rgb = result_np[:, :, :3]
            alpha = result_np[:, :, 3:4]
            white_bg = np.ones_like(rgb)
            result_np = rgb * alpha + white_bg * (1 - alpha)

        result_tensor = torch.from_numpy(result_np).unsqueeze(0)

        print(f"[3DSprite] Pixelated: {original_size} -> {pixel_resolution}x{pixel_resolution} -> {output_size}x{output_size}, {color_count} colors")

        return (result_tensor,)


class CombineImages8:
    """
    Combines 8 images into a single sprite sheet or grid.
    """

    CATEGORY = "3DSprite"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("sprite_sheet",)
    FUNCTION = "combine"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_N": ("IMAGE",),
                "image_NE": ("IMAGE",),
                "image_E": ("IMAGE",),
                "image_SE": ("IMAGE",),
                "image_S": ("IMAGE",),
                "image_SW": ("IMAGE",),
                "image_W": ("IMAGE",),
                "image_NW": ("IMAGE",),
                "layout": (["2x4", "4x2", "1x8", "8x1"],),
            }
        }

    def combine(
        self,
        image_N: torch.Tensor,
        image_NE: torch.Tensor,
        image_E: torch.Tensor,
        image_SE: torch.Tensor,
        image_S: torch.Tensor,
        image_SW: torch.Tensor,
        image_W: torch.Tensor,
        image_NW: torch.Tensor,
        layout: str
    ):
        images = [image_N, image_NE, image_E, image_SE, image_S, image_SW, image_W, image_NW]

        _, h, w, c = images[0].shape

        layouts = {
            "2x4": (2, 4),
            "4x2": (4, 2),
            "1x8": (1, 8),
            "8x1": (8, 1),
        }
        rows, cols = layouts[layout]

        output_h = rows * h
        output_w = cols * w
        output = torch.zeros((1, output_h, output_w, c), dtype=images[0].dtype)

        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            y_start = row * h
            x_start = col * w
            output[0, y_start:y_start+h, x_start:x_start+w, :] = img[0]

        print(f"[3DSprite] Combined 8 images into {layout} sprite sheet ({output_w}x{output_h})")

        return (output,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "RenderMesh8Directions": RenderMesh8Directions,
    "PixelateImage": PixelateImage,
    "CombineImages8": CombineImages8,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RenderMesh8Directions": "Render Mesh 8 Directions",
    "PixelateImage": "Pixelate Image",
    "CombineImages8": "Combine 8 Images",
}
