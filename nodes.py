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

def _check_torchmcubes():
    """Check that native torchmcubes is available. No fallbacks - they don't work correctly."""
    try:
        import torchmcubes
        print("[TripoSR] torchmcubes found and available")
    except ImportError:
        print("[TripoSR] ERROR: torchmcubes not installed!")
        print("[TripoSR] Install it with: pip install git+https://github.com/tatsy/torchmcubes.git")
        print("[TripoSR] Note: Requires CUDA toolkit and matching PyTorch CUDA version")
        raise ImportError(
            "torchmcubes is required for TripoSR mesh extraction. "
            "Install with: pip install git+https://github.com/tatsy/torchmcubes.git"
        )

# Apply patches BEFORE any TripoSR imports
_patch_rembg()
_check_torchmcubes()
# ============================================================================


# Global model cache to avoid reloading
_triposr_model_cache = {}


def resize_foreground(image: Image.Image, ratio: float = 0.85) -> Image.Image:
    """
    Resize foreground using alpha channel to find object bounds.
    This is the crucial preprocessing step from TripoSR's utils.

    Args:
        image: PIL Image in RGBA mode with transparent background
        ratio: Target occupancy ratio (0.85 = object fills 85% of the image)

    Returns:
        RGBA image with foreground centered and padded
    """
    image_np = np.array(image)

    if image_np.shape[-1] != 4:
        print("[TripoSR] Warning: resize_foreground expects RGBA image, got", image_np.shape)
        return image

    # Find non-transparent pixels using alpha channel
    alpha = np.where(image_np[..., 3] > 0)

    if len(alpha[0]) == 0:
        print("[TripoSR] Warning: No foreground pixels found (all transparent)")
        return image

    # Extract bounding box of foreground
    y1, y2 = alpha[0].min(), alpha[0].max()
    x1, x2 = alpha[1].min(), alpha[1].max()

    print(f"[TripoSR] Foreground bounds: x=[{x1}, {x2}], y=[{y1}, {y2}]")

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

    print(f"[TripoSR] Resized foreground: {new_image.shape[0]}x{new_image.shape[1]}")
    return Image.fromarray(new_image)


def rgba_to_rgb_gray_background(image: Image.Image) -> Image.Image:
    """
    Convert RGBA to RGB by compositing with gray (0.5) background.
    This matches TripoSR's expected preprocessing.

    Args:
        image: PIL Image in RGBA mode

    Returns:
        RGB image with gray background where transparency was
    """
    image_np = np.array(image).astype(np.float32) / 255.0

    if image_np.shape[-1] != 4:
        # Already RGB or other format
        return image.convert('RGB')

    rgb = image_np[:, :, :3]
    alpha = image_np[:, :, 3:4]

    # Composite with 0.5 gray background (TripoSR default)
    composited = rgb * alpha + (1 - alpha) * 0.5

    result = (composited * 255.0).astype(np.uint8)
    return Image.fromarray(result, 'RGB')


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

    IMPORTANT: For best results, provide an RGBA image with transparent background.
    Use ComfyUI's background removal nodes before this node.

    The preprocessing pipeline:
    1. resize_foreground() - Uses alpha channel to find the object's bounding box,
       crops to just the foreground, and centers it with proper padding (85% occupancy)
    2. rgba_to_rgb_gray_background() - Composites RGBA to RGB with 0.5 gray background,
       which is how TripoSR expects its input

    Without proper alpha channel, TripoSR cannot distinguish foreground from background,
    resulting in flat "relief" meshes instead of proper 3D.
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
                "unload_model": ("BOOLEAN", {"default": True}),
                "use_texture": ("BOOLEAN", {"default": True}),
                "texture_resolution": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 256}),
            }
        }

    def generate(self, model, image: torch.Tensor, resolution: int, unload_model: bool, use_texture: bool, texture_resolution: int):
        global _triposr_model_cache

        # Free up VRAM by unloading ComfyUI models (Flux, etc.) before TripoSR inference
        try:
            import comfy.model_management as mm
            print("[TripoSR] Unloading ComfyUI models to free VRAM...")
            mm.unload_all_models()
            mm.soft_empty_cache()
            print("[TripoSR] ComfyUI models unloaded")
        except Exception as e:
            print(f"[TripoSR] Could not unload ComfyUI models: {e}")

        # Convert ComfyUI IMAGE tensor (B, H, W, C) to PIL Image
        # ComfyUI images are float32 [0, 1]
        img_np = image[0].cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        # Handle RGBA images (from ComfyUI background removal)
        if img_np.shape[2] == 4:
            pil_image = Image.fromarray(img_np, 'RGBA')
            print(f"[TripoSR] Input: RGBA image {pil_image.size}")

            # CRUCIAL: Use TripoSR's preprocessing pipeline
            # Step 1: Resize foreground using alpha channel to find and center object
            pil_image = resize_foreground(pil_image, ratio=0.85)
            print(f"[TripoSR] After resize_foreground: {pil_image.size}")

            # Step 2: Convert RGBA to RGB with gray background (TripoSR's default)
            pil_image = rgba_to_rgb_gray_background(pil_image)
            print(f"[TripoSR] After RGBA->RGB: {pil_image.size}, mode={pil_image.mode}")
        else:
            pil_image = Image.fromarray(img_np, 'RGB')
            print(f"[TripoSR] Input: RGB image {pil_image.size} (no alpha channel)")
            print("[TripoSR] Warning: For best results, use an RGBA image with transparent background")

        # Ensure image is RGB (TripoSR expects RGB)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Get model device
        device = next(model.parameters()).device

        print(f"[TripoSR] Generating 3D mesh at resolution {resolution}...")

        # Run TripoSR inference
        with torch.no_grad():
            scene_codes = model([pil_image], device=device)

            if use_texture:
                # Use texture baking for better color accuracy
                print(f"[TripoSR] Using texture baking at {texture_resolution}x{texture_resolution}...")
                # Extract mesh without vertex colors first
                meshes = model.extract_mesh(
                    scene_codes,
                    has_vertex_color=False,
                    resolution=resolution
                )
                mesh = meshes[0]

                # Bake texture
                try:
                    from tsr.bake_texture import bake_texture
                    import xatlas

                    print("[TripoSR] Baking texture...")
                    bake_output = bake_texture(mesh, model, scene_codes[0], texture_resolution)

                    # Create new mesh with UV coordinates
                    new_vertices = mesh.vertices[bake_output["vmapping"]]
                    new_faces = bake_output["indices"]
                    uvs = bake_output["uvs"]

                    # Convert colors to float for processing
                    texture_colors = bake_output["colors"].copy()  # RGBA float [0,1]

                    # Apply color correction to fix washed out TripoSR colors
                    rgb = texture_colors[:, :, :3]
                    alpha = texture_colors[:, :, 3:4]

                    # Only process non-transparent pixels
                    mask = alpha[:, :, 0] > 0.01

                    if mask.any():
                        # 1. Auto-levels: stretch RGB range to use full 0-1 range
                        rgb_masked = rgb[mask]
                        for c in range(3):
                            channel = rgb_masked[:, c]
                            c_min, c_max = np.percentile(channel, [1, 99])
                            if c_max > c_min:
                                rgb[:, :, c] = np.clip((rgb[:, :, c] - c_min) / (c_max - c_min), 0, 1)

                        # 2. Boost saturation
                        gray = np.mean(rgb, axis=2, keepdims=True)
                        saturation_boost = 1.3
                        rgb = gray + (rgb - gray) * saturation_boost
                        rgb = np.clip(rgb, 0, 1)

                        # 3. Increase contrast (S-curve)
                        contrast = 1.2
                        rgb = (rgb - 0.5) * contrast + 0.5
                        rgb = np.clip(rgb, 0, 1)

                    # Recombine
                    texture_colors = np.concatenate([rgb, alpha], axis=2)
                    texture_colors = (texture_colors * 255.0).astype(np.uint8)

                    # Flip texture vertically (UV convention)
                    texture_image = Image.fromarray(texture_colors).transpose(Image.FLIP_TOP_BOTTOM)
                    print("[TripoSR] Applied color correction to texture")

                    # Create textured mesh
                    mesh = trimesh.Trimesh(
                        vertices=new_vertices,
                        faces=new_faces,
                    )

                    # Apply texture using trimesh's TextureVisuals
                    from trimesh.visual.texture import TextureVisuals
                    from trimesh.visual.material import SimpleMaterial

                    material = SimpleMaterial(image=texture_image)
                    mesh.visual = TextureVisuals(uv=uvs, image=texture_image, material=material)

                    print(f"[TripoSR] Texture baked successfully: {texture_resolution}x{texture_resolution}")
                except Exception as e:
                    print(f"[TripoSR] Texture baking failed: {e}, falling back to vertex colors")
                    meshes = model.extract_mesh(
                        scene_codes,
                        has_vertex_color=True,
                        resolution=resolution
                    )
                    mesh = meshes[0]
            else:
                # Use vertex colors (faster but less accurate colors)
                meshes = model.extract_mesh(
                    scene_codes,
                    has_vertex_color=True,
                    resolution=resolution
                )
                mesh = meshes[0]

        print(f"[TripoSR] Mesh generated: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Log mesh bounds for debugging
        if len(mesh.vertices) > 0:
            verts = mesh.vertices
            print(f"[TripoSR] Mesh bounds: X=[{verts[:,0].min():.3f}, {verts[:,0].max():.3f}], "
                  f"Y=[{verts[:,1].min():.3f}, {verts[:,1].max():.3f}], "
                  f"Z=[{verts[:,2].min():.3f}, {verts[:,2].max():.3f}]")

        # Unload model to free VRAM for other models (e.g., Flux)
        if unload_model:
            print("[TripoSR] Unloading model to free VRAM...")
            model.to("cpu")
            _triposr_model_cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[TripoSR] Model unloaded, VRAM freed")

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
        # Create pyrender scene - higher ambient for flatter, more "unlit" look
        # This preserves vertex colors better without harsh lighting
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=[0.6, 0.6, 0.6])

        # Convert trimesh to pyrender mesh
        # DON'T pass a material - let pyrender use vertex colors from trimesh
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene.add(pyrender_mesh)

        # Add camera with wider FOV for better framing
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        camera_pose = self._create_camera_pose(azimuth, elevation, distance)
        scene.add(camera, pose=camera_pose)

        # Soft directional lights - lower intensity to not wash out vertex colors
        # Key light (main)
        key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
        key_pose = self._create_camera_pose(azimuth - 30, elevation + 30, 1.0)
        scene.add(key_light, pose=key_pose)

        # Fill light (opposite side) - very soft
        fill_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.4)
        fill_pose = self._create_camera_pose(azimuth + 60, elevation, 1.0)
        scene.add(fill_light, pose=fill_pose)

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
        background_color: str,
        auto_align: bool,
        mesh_pitch: float,
        mesh_roll: float
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

        if auto_align:
            # Auto-align using PCA to find principal axes
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=3)
                pca.fit(mesh_centered.vertices)

                # Create rotation matrix from principal components
                # PCA components are ordered by variance (largest first)
                components = pca.components_

                # We want: X = width (largest horizontal), Y = up, Z = depth
                # Reorder so the smallest variance axis becomes Y (up)
                variances = pca.explained_variance_
                up_idx = np.argmin(variances)  # Smallest variance = thinnest direction = up

                # Build rotation matrix
                axes = list(range(3))
                axes.remove(up_idx)

                rotation = np.eye(4)
                rotation[:3, 1] = components[up_idx]  # Y = up (smallest variance)
                rotation[:3, 0] = components[axes[0]]  # X = first horizontal
                rotation[:3, 2] = components[axes[1]]  # Z = second horizontal

                # Ensure right-handed coordinate system
                if np.linalg.det(rotation[:3, :3]) < 0:
                    rotation[:3, 2] *= -1

                mesh_centered.apply_transform(rotation.T)  # Inverse rotation

                # Rotate 180° around Y to face correct direction
                rot_180 = trimesh.transformations.rotation_matrix(
                    np.pi, [0, 1, 0], point=[0, 0, 0]
                )
                mesh_centered.apply_transform(rot_180)
                print("[TripoSR] Auto-aligned mesh using PCA")
            except Exception as e:
                print(f"[TripoSR] Auto-align failed: {e}, using manual rotation")
                # Fallback to manual rotation
                rot_x = trimesh.transformations.rotation_matrix(
                    -np.pi / 2, [1, 0, 0], point=[0, 0, 0]
                )
                rot_y = trimesh.transformations.rotation_matrix(
                    np.pi / 2, [0, 1, 0], point=[0, 0, 0]
                )
                mesh_centered.apply_transform(rot_x)
                mesh_centered.apply_transform(rot_y)
        else:
            # Manual rotation to stand upright and face forward
            rot_x = trimesh.transformations.rotation_matrix(
                -np.pi / 2, [1, 0, 0], point=[0, 0, 0]
            )
            rot_y = trimesh.transformations.rotation_matrix(
                -np.pi / 2, [0, 1, 0], point=[0, 0, 0]  # Changed from +90 to -90
            )
            mesh_centered.apply_transform(rot_x)
            mesh_centered.apply_transform(rot_y)

        # Apply user pitch/roll correction to fix tilt from source image perspective
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

        # Scale to fit in unit cube, larger factor = fills more of frame
        bounds = mesh_centered.bounds  # Recalculate after rotation
        extents = bounds[1] - bounds[0]  # [width, height, depth]
        max_extent = np.max(extents)
        scale = 1.0 / max_extent
        mesh_centered.vertices *= scale * 1.2  # Fill more of the frame

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
