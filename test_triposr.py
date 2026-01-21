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

# Stub torchmcubes with PyMCubes or scikit-image
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

print("Loading model...")
model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt"
)
model.to("cuda:0")

print("Loading image...")
image = Image.open("TripoSR/examples/chair.png")

# Convert RGBA to RGB with white background
if image.mode == 'RGBA':
    background = Image.new('RGB', image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[3])
    image = background
    print(f"Converted RGBA to RGB, size: {image.size}")
elif image.mode != 'RGB':
    image = image.convert('RGB')
    print(f"Converted {image.mode} to RGB")

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

mesh.export("/tmp/chair_test.obj")
print("Exported to /tmp/chair_test.obj")
