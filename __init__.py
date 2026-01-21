"""
ComfyUI 3D Sprite Generator

A ComfyUI custom node package for converting images to 3D meshes
and rendering them from 8 cardinal/intercardinal directions.

Supported models:
- TripoSR: Fast image-to-3D
- Hunyuan3D: High-quality image-to-3D

Nodes:
- Load TripoSR Model / Load Hunyuan3D Model
- Image to 3D Mesh (TripoSR / Hunyuan3D)
- Render Mesh 8 Directions
- Pixelate Image
- Combine 8 Images
"""

# Import node mappings from all modules
from .nodes_common import NODE_CLASS_MAPPINGS as COMMON_MAPPINGS
from .nodes_common import NODE_DISPLAY_NAME_MAPPINGS as COMMON_DISPLAY_MAPPINGS

# Try to import TripoSR nodes (may fail if torchmcubes not installed)
try:
    from .nodes_triposr import NODE_CLASS_MAPPINGS as TRIPOSR_MAPPINGS
    from .nodes_triposr import NODE_DISPLAY_NAME_MAPPINGS as TRIPOSR_DISPLAY_MAPPINGS
except ImportError as e:
    print(f"[3DSprite] TripoSR nodes not available: {e}")
    TRIPOSR_MAPPINGS = {}
    TRIPOSR_DISPLAY_MAPPINGS = {}

# Try to import Hunyuan3D nodes (may fail if hy3dgen not installed)
try:
    from .nodes_hunyuan3d import NODE_CLASS_MAPPINGS as HUNYUAN_MAPPINGS
    from .nodes_hunyuan3d import NODE_DISPLAY_NAME_MAPPINGS as HUNYUAN_DISPLAY_MAPPINGS
except ImportError as e:
    print(f"[3DSprite] Hunyuan3D nodes not available: {e}")
    HUNYUAN_MAPPINGS = {}
    HUNYUAN_DISPLAY_MAPPINGS = {}

# Merge all mappings
NODE_CLASS_MAPPINGS = {
    **COMMON_MAPPINGS,
    **TRIPOSR_MAPPINGS,
    **HUNYUAN_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **COMMON_DISPLAY_MAPPINGS,
    **TRIPOSR_DISPLAY_MAPPINGS,
    **HUNYUAN_DISPLAY_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

WEB_DIRECTORY = None
