#!/usr/bin/env python3
"""Test TripoSR directly to verify mesh generation."""

import sys
sys.path.insert(0, 'TripoSR')

# Stub rembg
import types
rembg = types.ModuleType('rembg')
rembg.remove = lambda x, **kw: x
rembg.new_session = lambda *a, **kw: None
sys.modules['rembg'] = rembg

import torch
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
