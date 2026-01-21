"""
ComfyUI nodes for TripoSR-based 8-directional sprite rotation.

Provides nodes for:
- Loading TripoSR model
- Converting images to 3D meshes
- Rendering meshes from 8 cardinal/intercardinal directions
- Pixelating rendered images for pixel art style
"""

import os
import sys
from pathlib import Path

# Add TripoSR to path if cloned locally
_triposr_local = Path(__file__).parent / "TripoSR"
if _triposr_local.exists() and str(_triposr_local) not in sys.path:
    sys.path.insert(0, str(_triposr_local))

import torch
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional
import trimesh

# Set headless rendering platform before importing pyrender
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

import pyrender


# ============================================================================
# Patch missing modules before importing TripoSR
# ============================================================================
import types

def _patch_rembg():
    """Create a fake rembg module (user should use ComfyUI's background removal)."""
    try:
        import rembg
        return  # Already available
    except ImportError:
        pass

    print("[TripoSR] rembg not found, creating stub (use ComfyUI background removal instead)")

    def remove(image, session=None, **kwargs):
        """Stub - returns image unchanged. Use ComfyUI's background removal nodes."""
        print("[TripoSR] Warning: rembg.remove() called but rembg not installed. Image unchanged.")
        return image

    def new_session(*args, **kwargs):
        """Stub session creator."""
        return None

    # Create fake module
    fake_rembg = types.ModuleType('rembg')
    fake_rembg.remove = remove
    fake_rembg.new_session = new_session
    sys.modules['rembg'] = fake_rembg

def _patch_torchmcubes():
    """Create a fake torchmcubes module using mcubes or scikit-image."""
    try:
        import torchmcubes
        print("[TripoSR] torchmcubes found and available")
        return  # Already available, no patch needed
    except ImportError:
        pass

    # Try PyMCubes first (closer to torchmcubes behavior)
    try:
        import mcubes
        print("[TripoSR] Using PyMCubes as torchmcubes fallback")

        def marching_cubes(volume, threshold):
            device = volume.device if isinstance(volume, torch.Tensor) else torch.device('cpu')

            if isinstance(volume, torch.Tensor):
                vol_np = volume.detach().cpu().numpy()
            else:
                vol_np = np.array(volume)

            print(f"[TripoSR] mcubes input: shape={vol_np.shape}, min={vol_np.min():.4f}, max={vol_np.max():.4f}, threshold={threshold}")

            # PyMCubes marching_cubes
            verts, faces = mcubes.marching_cubes(vol_np, threshold)

            print(f"[TripoSR] mcubes output: {len(verts)} vertices, {len(faces)} faces")
            if len(verts) > 0:
                print(f"[TripoSR] Vertex bounds: X=[{verts[:,0].min():.2f}, {verts[:,0].max():.2f}], Y=[{verts[:,1].min():.2f}, {verts[:,1].max():.2f}], Z=[{verts[:,2].min():.2f}, {verts[:,2].max():.2f}]")

            verts_tensor = torch.from_numpy(verts.astype(np.float32)).to(device)
            faces_tensor = torch.from_numpy(faces.astype(np.int64)).to(device)

            return verts_tensor, faces_tensor

        fake_module = types.ModuleType('torchmcubes')
        fake_module.marching_cubes = marching_cubes
        sys.modules['torchmcubes'] = fake_module
        return

    except ImportError:
        pass

    # Fallback to scikit-image
    print("[TripoSR] Using scikit-image marching cubes as torchmcubes fallback")
    from skimage import measure

    def marching_cubes(volume, threshold):
        device = volume.device if isinstance(volume, torch.Tensor) else torch.device('cpu')

        if isinstance(volume, torch.Tensor):
            vol_np = volume.detach().cpu().numpy()
        else:
            vol_np = np.array(volume)

        if vol_np.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {vol_np.shape}")

        print(f"[TripoSR] skimage input: shape={vol_np.shape}, min={vol_np.min():.4f}, max={vol_np.max():.4f}, threshold={threshold}")

        try:
            verts, faces, normals, values = measure.marching_cubes(vol_np, level=threshold)
        except Exception as e:
            print(f"[TripoSR] Marching cubes failed: {e}")
            return torch.zeros((0, 3), device=device), torch.zeros((0, 3), dtype=torch.long, device=device)

        print(f"[TripoSR] skimage output: {len(verts)} vertices, {len(faces)} faces")
        if len(verts) > 0:
            print(f"[TripoSR] Vertex bounds: X=[{verts[:,0].min():.2f}, {verts[:,0].max():.2f}], Y=[{verts[:,1].min():.2f}, {verts[:,1].max():.2f}], Z=[{verts[:,2].min():.2f}, {verts[:,2].max():.2f}]")

        verts_tensor = torch.from_numpy(verts.copy().astype(np.float32)).to(device)
        faces_tensor = torch.from_numpy(faces.copy().astype(np.int64)).to(device)

        return verts_tensor, faces_tensor

    fake_module = types.ModuleType('torchmcubes')
    fake_module.marching_cubes = marching_cubes
    sys.modules['torchmcubes'] = fake_module

# Apply patches BEFORE any TripoSR imports
_patch_rembg()
_patch_torchmcubes()
# ============================================================================


# Global model cache to avoid reloading
_triposr_model_cache = {}


def get_triposr_checkpoints():
    """Get list of available TripoSR checkpoints from models directory."""
    import folder_paths

    # Register TripoSR model folder if not exists
    if "triposr" not in folder_paths.folder_names_and_paths:
        triposr_path = os.path.join(folder_paths.models_dir, "triposr")
        os.makedirs(triposr_path, exist_ok=True)
        folder_paths.folder_names_and_paths["triposr"] = ([triposr_path], {".ckpt", ".safetensors", ".pt", ".pth"})

    checkpoints = folder_paths.get_filename_list("triposr")
    if not checkpoints:
        checkpoints = ["model.ckpt"]  # Default name
    return checkpoints


class LoadTripoSRModel:
    """
    Loads the TripoSR model from a local checkpoint file.
    Model is cached for reuse across multiple generations.

    Place checkpoint files in: ComfyUI/models/triposr/
    Required files:
    - model.ckpt (or your checkpoint name)
    - config.yaml (TripoSR config)
    """

    CATEGORY = "TripoSR"
    RETURN_TYPES = ("TRIPOSR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": (get_triposr_checkpoints(),),
                "device": (["cuda:0", "cuda:1", "cpu"],),
            }
        }

    def load(self, checkpoint: str, device: str):
        import folder_paths
        global _triposr_model_cache

        cache_key = f"{checkpoint}_{device}"

        if cache_key in _triposr_model_cache:
            print(f"[TripoSR] Using cached model: {checkpoint} on {device}")
            return (_triposr_model_cache[cache_key],)

        print(f"[TripoSR] Loading model from checkpoint: {checkpoint} on {device}...")

        # Check if TripoSR is available
        triposr_path = Path(__file__).parent / "TripoSR"
        if not triposr_path.exists():
            raise ImportError(
                f"TripoSR not found at {triposr_path}\n"
                "Please clone it with:\n"
                f"  cd {Path(__file__).parent}\n"
                "  git clone https://github.com/VAST-AI-Research/TripoSR.git"
            )

        # Ensure path is in sys.path
        if str(triposr_path) not in sys.path:
            sys.path.insert(0, str(triposr_path))

        try:
            from tsr.system import TSR
        except ImportError as e:
            raise ImportError(
                f"Failed to import TripoSR: {e}\n"
                f"TripoSR path: {triposr_path}\n"
                f"sys.path includes TripoSR: {str(triposr_path) in sys.path}\n"
                "Try installing TripoSR dependencies:\n"
                "  uv pip install transformers einops omegaconf huggingface_hub"
            )

        # Get the full path to the checkpoint
        checkpoint_path = folder_paths.get_full_path("triposr", checkpoint)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Checkpoint '{checkpoint}' not found. "
                f"Place your TripoSR checkpoint in: ComfyUI/models/triposr/"
            )

        # Get the directory containing the checkpoint
        model_dir = os.path.dirname(checkpoint_path)
        checkpoint_name = os.path.basename(checkpoint_path)

        # Check for config.yaml in the same directory
        config_path = os.path.join(model_dir, "config.yaml")
        if not os.path.exists(config_path):
            # Try to find config.yaml in parent directory or create default
            parent_config = os.path.join(os.path.dirname(model_dir), "config.yaml")
            if os.path.exists(parent_config):
                config_path = parent_config
            else:
                print(f"[TripoSR] Warning: config.yaml not found at {config_path}")
                print("[TripoSR] Attempting to load with default config...")

        # Load model from local path
        model = TSR.from_pretrained(
            model_dir,
            config_name="config.yaml",
            weight_name=checkpoint_name
        )
        model.to(device)
        model.renderer.set_chunk_size(8192)  # Reduce memory usage

        _triposr_model_cache[cache_key] = model
        print(f"[TripoSR] Model loaded successfully from {checkpoint_path} on {device}")

        return (model,)


class ImageTo3DMesh:
    """
    Converts an input image to a 3D mesh using TripoSR.

    Note: Use ComfyUI's background removal nodes before this node
    for best results with TripoSR.
    """

    CATEGORY = "TripoSR"
    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("TRIPOSR_MODEL",),
                "image": ("IMAGE",),
                "resolution": ("INT", {"default": 256, "min": 64, "max": 512, "step": 32}),
            }
        }

    def generate(self, model, image: torch.Tensor, resolution: int):
        # Convert ComfyUI IMAGE tensor (B, H, W, C) to PIL Image
        # ComfyUI images are float32 [0, 1]
        img_np = image[0].cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        # Handle RGBA images (from ComfyUI background removal)
        if img_np.shape[2] == 4:
            pil_image = Image.fromarray(img_np, 'RGBA')
            # Create white background for transparent areas
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            background.paste(pil_image, mask=pil_image.split()[3])
            pil_image = background
        else:
            pil_image = Image.fromarray(img_np, 'RGB')

        # Ensure image is RGB (TripoSR expects RGB)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Get model device
        device = next(model.parameters()).device

        print(f"[TripoSR] Generating 3D mesh at resolution {resolution}...")

        # Run TripoSR inference
        with torch.no_grad():
            scene_codes = model([pil_image], device=device)
            meshes = model.extract_mesh(
                scene_codes,
                has_vertex_color=True,
                resolution=resolution
            )

        mesh = meshes[0]
        print(f"[TripoSR] Mesh generated: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        return (mesh,)


class RenderMesh8Directions:
    """
    Renders a mesh from 8 cardinal and intercardinal directions.
    Returns 8 separate images: N, NE, E, SE, S, SW, W, NW.
    """

    CATEGORY = "TripoSR"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    FUNCTION = "render"

    # Direction angles (azimuth in degrees, clockwise from front)
    DIRECTIONS = [
        ("N", 0),      # Front
        ("NE", 45),    # Front-right
        ("E", 90),     # Right
        ("SE", 135),   # Back-right
        ("S", 180),    # Back
        ("SW", 225),   # Back-left
        ("W", 270),    # Left
        ("NW", 315),   # Front-left
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "render_size": ("INT", {"default": 512, "min": 64, "max": 1024, "step": 64}),
                "elevation": ("FLOAT", {"default": 20.0, "min": -90.0, "max": 90.0, "step": 5.0}),
                "distance": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.1}),
                "background_color": (["white", "black", "transparent"],),
            }
        }

    def _create_camera_pose(self, azimuth_deg: float, elevation_deg: float, distance: float) -> np.ndarray:
        """Create camera pose matrix for given azimuth and elevation angles."""
        azimuth = np.radians(azimuth_deg)
        elevation = np.radians(elevation_deg)

        # Camera position in spherical coordinates
        x = distance * np.cos(elevation) * np.sin(azimuth)
        y = distance * np.sin(elevation)
        z = distance * np.cos(elevation) * np.cos(azimuth)

        camera_pos = np.array([x, y, z])

        # Look at origin
        target = np.array([0, 0, 0])
        up = np.array([0, 1, 0])

        # Create look-at matrix
        forward = target - camera_pos
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        # Camera pose matrix (camera-to-world)
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
        # Create pyrender scene with more ambient light to show vertex colors
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=[0.7, 0.7, 0.7])

        # Convert trimesh to pyrender mesh
        # DON'T pass a material - let pyrender use vertex colors from trimesh
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene.add(pyrender_mesh)

        # Add camera with wider FOV for better framing
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        camera_pose = self._create_camera_pose(azimuth, elevation, distance)
        scene.add(camera, pose=camera_pose)

        # Add softer lights that don't wash out vertex colors
        # Key light (main)
        key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5)
        key_pose = self._create_camera_pose(azimuth - 45, elevation + 30, 1.0)
        scene.add(key_light, pose=key_pose)

        # Fill light (softer, opposite side)
        fill_light = pyrender.DirectionalLight(color=[0.9, 0.9, 0.9], intensity=0.8)
        fill_pose = self._create_camera_pose(azimuth + 45, elevation, 1.0)
        scene.add(fill_light, pose=fill_pose)

        # Back light (rim)
        back_light = pyrender.DirectionalLight(color=[0.8, 0.8, 0.8], intensity=0.5)
        back_pose = self._create_camera_pose(azimuth + 180, elevation - 10, 1.0)
        scene.add(back_light, pose=back_pose)

        # Render
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
        background_color: str
    ):
        # Set background color
        bg_colors = {
            "white": (1.0, 1.0, 1.0, 1.0),
            "black": (0.0, 0.0, 0.0, 1.0),
            "transparent": (0.0, 0.0, 0.0, 0.0),
        }
        bg_color = bg_colors.get(background_color, (1.0, 1.0, 1.0, 1.0))

        # Center the mesh at origin using bounding box center
        mesh_centered = mesh.copy()
        bounds = mesh_centered.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        center = (bounds[0] + bounds[1]) / 2
        mesh_centered.vertices -= center

        # Scale to fit in unit cube, then apply 0.7 factor for padding
        extents = bounds[1] - bounds[0]  # [width, height, depth]
        max_extent = np.max(extents)
        scale = 1.0 / max_extent
        mesh_centered.vertices *= scale * 0.7

        rendered_images = []

        for direction_name, azimuth in self.DIRECTIONS:
            print(f"[TripoSR] Rendering {direction_name} ({azimuth}deg)...")

            color = self._render_single_view(
                mesh_centered,
                azimuth,
                elevation,
                distance,
                render_size,
                bg_color
            )

            # Convert to ComfyUI IMAGE format (B, H, W, C) float32 [0, 1]
            img_tensor = torch.from_numpy(color.astype(np.float32) / 255.0)
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

            rendered_images.append(img_tensor)

        print(f"[TripoSR] All 8 directions rendered at {render_size}x{render_size}")

        return tuple(rendered_images)


class PixelateImage:
    """
    Pixelates an image with optional color quantization.
    Useful for creating pixel art style sprites from rendered meshes.
    """

    CATEGORY = "TripoSR"
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
        # Convert to numpy
        img_array = np.array(img)
        original_shape = img_array.shape

        # Handle RGBA
        has_alpha = img_array.shape[2] == 4 if len(img_array.shape) > 2 else False

        if has_alpha:
            rgb = img_array[:, :, :3]
            alpha = img_array[:, :, 3]
        else:
            rgb = img_array
            alpha = None

        # Reshape to (N, 3)
        pixels = rgb.reshape(-1, 3).astype(np.float32)

        # K-means clustering using scikit-learn or manual implementation
        try:
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, n_init=3)
            labels = kmeans.fit_predict(pixels)
            centers = kmeans.cluster_centers_
        except ImportError:
            # Fallback to PIL quantize
            print("[TripoSR] sklearn not available, using PIL quantize")
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

        # Reconstruct image
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
        # Convert ComfyUI IMAGE tensor to PIL
        img_np = image[0].cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        # Check if we have alpha channel
        if img_np.shape[2] == 4:
            pil_image = Image.fromarray(img_np, 'RGBA')
        else:
            pil_image = Image.fromarray(img_np, 'RGB')

        original_size = pil_image.size

        # Downscale to pixel resolution using nearest neighbor
        small = pil_image.resize(
            (pixel_resolution, pixel_resolution),
            Image.Resampling.NEAREST
        )

        # Quantize colors
        quantized = self._quantize_colors_kmeans(small, color_count)

        # Upscale back to output size using nearest neighbor (keeps pixelated look)
        result = quantized.resize(
            (output_size, output_size),
            Image.Resampling.NEAREST
        )

        # Convert back to ComfyUI format
        result_np = np.array(result).astype(np.float32) / 255.0

        # Ensure 3 channels (RGB) for ComfyUI
        if result_np.shape[2] == 4:
            # Composite RGBA onto white background
            rgb = result_np[:, :, :3]
            alpha = result_np[:, :, 3:4]
            white_bg = np.ones_like(rgb)
            result_np = rgb * alpha + white_bg * (1 - alpha)

        result_tensor = torch.from_numpy(result_np).unsqueeze(0)

        print(f"[TripoSR] Pixelated: {original_size} -> {pixel_resolution}x{pixel_resolution} -> {output_size}x{output_size}, {color_count} colors")

        return (result_tensor,)


class CombineImages8:
    """
    Combines 8 images into a single sprite sheet or grid.
    Useful for exporting all rotation directions as one image.
    """

    CATEGORY = "TripoSR"
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

        # Get dimensions from first image
        _, h, w, c = images[0].shape

        # Parse layout
        layouts = {
            "2x4": (2, 4),  # 2 rows, 4 columns
            "4x2": (4, 2),  # 4 rows, 2 columns
            "1x8": (1, 8),  # 1 row, 8 columns
            "8x1": (8, 1),  # 8 rows, 1 column
        }
        rows, cols = layouts[layout]

        # Create output tensor
        output_h = rows * h
        output_w = cols * w
        output = torch.zeros((1, output_h, output_w, c), dtype=images[0].dtype)

        # Place images in grid
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            y_start = row * h
            x_start = col * w
            output[0, y_start:y_start+h, x_start:x_start+w, :] = img[0]

        print(f"[TripoSR] Combined 8 images into {layout} sprite sheet ({output_w}x{output_h})")

        return (output,)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "LoadTripoSRModel": LoadTripoSRModel,
    "ImageTo3DMesh": ImageTo3DMesh,
    "RenderMesh8Directions": RenderMesh8Directions,
    "PixelateImage": PixelateImage,
    "CombineImages8": CombineImages8,
}

# Human-readable names for nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadTripoSRModel": "Load TripoSR Model",
    "ImageTo3DMesh": "Image to 3D Mesh",
    "RenderMesh8Directions": "Render Mesh 8 Directions",
    "PixelateImage": "Pixelate Image",
    "CombineImages8": "Combine 8 Images",
}
