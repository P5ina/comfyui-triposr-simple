"""
ComfyUI TripoSR Simple - 8-directional sprite rotation using TripoSR

A minimal ComfyUI custom node package for converting images to 3D meshes
using TripoSR and rendering them from 8 cardinal/intercardinal directions.

Installation:
1. Copy this folder to ComfyUI/custom_nodes/
2. Install dependencies: pip install -r requirements.txt
3. Restart ComfyUI

Nodes:
- Load TripoSR Model: Loads the TripoSR model (cached for reuse)
- Image to 3D Mesh: Converts an image to a 3D mesh
- Render Mesh 8 Directions: Renders mesh from N, NE, E, SE, S, SW, W, NW
- Pixelate Image: Downscales and quantizes colors for pixel art style
- Combine 8 Images: Combines 8 images into a sprite sheet
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

WEB_DIRECTORY = None
