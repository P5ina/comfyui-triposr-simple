# ComfyUI TripoSR Simple

A minimal ComfyUI custom node package for converting images to 3D meshes using TripoSR and rendering them from 8 cardinal/intercardinal directions.

## Installation

1. Copy this folder to `ComfyUI/custom_nodes/`
2. Run the install script:
   ```bash
   cd ComfyUI/custom_nodes/comfyui-triposr-simple
   python install.py
   ```
3. Download TripoSR model files to `ComfyUI/models/triposr/`:
   - `model.ckpt` - Main checkpoint (~1GB)
   - `config.yaml` - Model config

   Download from: https://huggingface.co/stabilityai/TripoSR

4. Restart ComfyUI

## Nodes

### Load TripoSR Model
Loads the TripoSR model from a local checkpoint file.

**Inputs:**
- `checkpoint`: Select from available checkpoints in `models/triposr/`
- `device`: cuda:0, cuda:1, or cpu

**Outputs:**
- `model`: TripoSR model instance (cached for reuse)

### Image to 3D Mesh
Converts an input image to a 3D mesh using TripoSR.

**Inputs:**
- `model`: TripoSR model from loader
- `image`: Input image (ComfyUI IMAGE)
- `resolution`: Mesh resolution (64-512, default 256)
- `remove_background`: Auto-remove background (requires rembg)

**Outputs:**
- `mesh`: Trimesh object with vertex colors

### Render Mesh 8 Directions
Renders the mesh from 8 cardinal and intercardinal directions.

**Inputs:**
- `mesh`: Mesh from ImageTo3DMesh
- `render_size`: Output image size (64-1024, default 512)
- `elevation`: Camera elevation angle (-90 to 90, default 20)
- `distance`: Camera distance (0.5-10, default 2.0)
- `background_color`: white, black, or transparent

**Outputs:**
- 8 images: N, NE, E, SE, S, SW, W, NW

### Pixelate Image
Pixelates rendered images with color quantization for pixel art style.

**Inputs:**
- `image`: Input image
- `pixel_resolution`: Downscale resolution (32, 64, 128, 256)
- `color_count`: Number of colors (8-256)
- `output_size`: Final output size (32-1024)

**Outputs:**
- `image`: Pixelated image

### Combine 8 Images
Combines 8 direction images into a sprite sheet.

**Inputs:**
- 8 images (N, NE, E, SE, S, SW, W, NW)
- `layout`: 2x4, 4x2, 1x8, or 8x1

**Outputs:**
- `sprite_sheet`: Combined image

## Example Workflows

### Regular Mode
```
LoadImage → LoadTripoSRModel → ImageTo3DMesh → RenderMesh8Directions → SaveImage
```

### Pixel Art Mode
```
LoadImage → LoadTripoSRModel → ImageTo3DMesh → RenderMesh8Directions → PixelateImage (x8) → CombineImages8 → SaveImage
```

## Troubleshooting

### Headless Rendering
If running on a server without display, ensure EGL or OSMesa is installed:
```bash
# Ubuntu/Debian
sudo apt-get install libegl1-mesa-dev libosmesa6-dev
```

Set environment variable if needed:
```bash
export PYOPENGL_PLATFORM=egl  # or osmesa
```

### Model Not Found
Ensure checkpoint files are in `ComfyUI/models/triposr/`:
- `model.ckpt`
- `config.yaml`

### CUDA Out of Memory
- Reduce mesh `resolution` (try 128 instead of 256)
- Reduce `render_size` (try 256 instead of 512)
- Use CPU for rendering if needed

## License

This package is MIT licensed. TripoSR model is also MIT licensed.
