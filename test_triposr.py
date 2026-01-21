#!/usr/bin/env python3
"""Test TripoSR directly to verify mesh generation."""

import sys
sys.path.insert(0, 'TripoSR')

import types
import numpy as np
import torch

# Stub rembg
rembg = types.ModuleType('rembg')
rembg.remove = lambda x, **kw: x
rembg.new_session = lambda *a, **kw: None
sys.modules['rembg'] = rembg

# Try to use native torchmcubes first, fall back to PyMCubes or scikit-image
try:
    import torchmcubes
    print("Using native torchmcubes (CUDA)")
except ImportError:
    print("Native torchmcubes not available, using fallback...")
    try:
        import mcubes
        print("Using PyMCubes for marching cubes")

        def marching_cubes(volume, threshold):
            device = volume.device if isinstance(volume, torch.Tensor) else torch.device('cpu')
            if isinstance(volume, torch.Tensor):
                vol_np = volume.detach().cpu().numpy()
            else:
                vol_np = np.array(volume)

            print(f"  mcubes input: shape={vol_np.shape}, min={vol_np.min():.4f}, max={vol_np.max():.4f}")
            verts, faces = mcubes.marching_cubes(vol_np, threshold)
            print(f"  mcubes output: {len(verts)} verts, {len(faces)} faces")

            return torch.from_numpy(verts.astype(np.float32)).to(device), torch.from_numpy(faces.astype(np.int64)).to(device)

    except ImportError:
        from skimage import measure
        print("Using scikit-image for marching cubes")

        def marching_cubes(volume, threshold):
            device = volume.device if isinstance(volume, torch.Tensor) else torch.device('cpu')
            if isinstance(volume, torch.Tensor):
                vol_np = volume.detach().cpu().numpy()
            else:
                vol_np = np.array(volume)

            print(f"  skimage input: shape={vol_np.shape}, min={vol_np.min():.4f}, max={vol_np.max():.4f}")
            verts, faces, _, _ = measure.marching_cubes(vol_np, level=threshold)
            print(f"  skimage output: {len(verts)} verts, {len(faces)} faces")

            return torch.from_numpy(verts.astype(np.float32)).to(device), torch.from_numpy(faces.astype(np.int64)).to(device)

    torchmcubes = types.ModuleType('torchmcubes')
    torchmcubes.marching_cubes = marching_cubes
    sys.modules['torchmcubes'] = torchmcubes

from PIL import Image
from tsr.system import TSR


def resize_foreground(image, ratio=0.85):
    """
    Resize foreground using alpha channel to find object bounds.
    This is the crucial preprocessing step from TripoSR.

    Args:
        image: PIL Image in RGBA mode
        ratio: Target occupancy ratio (0.85 = object fills 85% of the image)
    """
    image = np.array(image)
    assert image.shape[-1] == 4, "Image must be RGBA"

    # Find non-transparent pixels using alpha channel
    alpha = np.where(image[..., 3] > 0)

    if len(alpha[0]) == 0:
        print("Warning: No foreground pixels found!")
        return Image.fromarray(image)

    # Extract bounding box of foreground
    y1, y2 = alpha[0].min(), alpha[0].max()
    x1, x2 = alpha[1].min(), alpha[1].max()

    print(f"  Foreground bounds: x=[{x1}, {x2}], y=[{y1}, {y2}]")

    # Crop to foreground
    fg = image[y1:y2, x1:x2]

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

    print(f"  Resized foreground: {new_image.shape}")
    return Image.fromarray(new_image)


def rgba_to_rgb_with_gray_background(image):
    """
    Convert RGBA to RGB by compositing with gray (0.5) background.
    This is how TripoSR expects the input.
    """
    image_np = np.array(image).astype(np.float32) / 255.0

    rgb = image_np[:, :, :3]
    alpha = image_np[:, :, 3:4]

    # Composite with 0.5 gray background (TripoSR default)
    composited = rgb * alpha + (1 - alpha) * 0.5

    result = (composited * 255.0).astype(np.uint8)
    return Image.fromarray(result, 'RGB')


print("Loading model...")
model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt"
)
model.to("cuda:0")

print("Loading image...")
image = Image.open("TripoSR/examples/chair.png")
print(f"Original image: mode={image.mode}, size={image.size}")

# Convert to RGBA if needed
if image.mode != 'RGBA':
    image = image.convert('RGBA')
    print(f"Converted to RGBA")

# Step 1: Resize foreground using alpha channel (CRUCIAL STEP!)
print("Preprocessing: resize_foreground...")
image = resize_foreground(image, ratio=0.85)
image.save("/tmp/chair_preprocessed_rgba.png")
print(f"Saved preprocessed RGBA to /tmp/chair_preprocessed_rgba.png")

# Step 2: Convert RGBA to RGB with gray background
print("Converting RGBA to RGB with gray background...")
image = rgba_to_rgb_with_gray_background(image)
image.save("/tmp/chair_preprocessed_rgb.png")
print(f"Final RGB image: size={image.size}")

print("Running inference...")
with torch.no_grad():
    scene_codes = model([image], device="cuda:0")
    mesh = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=256)[0]

print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
print(f"Vertex bounds:")
print(f"  X: [{mesh.vertices[:,0].min():.3f}, {mesh.vertices[:,0].max():.3f}]")
print(f"  Y: [{mesh.vertices[:,1].min():.3f}, {mesh.vertices[:,1].max():.3f}]")
print(f"  Z: [{mesh.vertices[:,2].min():.3f}, {mesh.vertices[:,2].max():.3f}]")

# Check vertex colors
if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
    colors = mesh.visual.vertex_colors
    print(f"Vertex colors: shape={colors.shape}, min={colors.min()}, max={colors.max()}")
else:
    print("No vertex colors found!")

# Note: With native torchmcubes normals are correct, don't flip
# With PyMCubes fallback, normals may be inverted
# mesh.invert()

# Export to GLB (better vertex color support than OBJ)
mesh.export("/tmp/chair_test.glb")
print("Exported to /tmp/chair_test.glb")

# Also export to PLY (good vertex color support)
mesh.export("/tmp/chair_test.ply")
print("Exported to /tmp/chair_test.ply")
