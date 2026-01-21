"""
Hunyuan3D nodes for image-to-3D mesh generation.
"""

import os
import torch
import numpy as np
from PIL import Image

from .nodes_common import resize_foreground, rgba_to_rgb_gray_background


# Global model cache
_hunyuan3d_model_cache = {}
_hunyuan3d_texture_model_cache = {}


def get_hunyuan3d_checkpoints():
    """Get list of available Hunyuan3D checkpoints."""
    import folder_paths

    if "hunyuan3d" not in folder_paths.folder_names_and_paths:
        hunyuan3d_path = os.path.join(folder_paths.models_dir, "hunyuan3d")
        os.makedirs(hunyuan3d_path, exist_ok=True)
        folder_paths.folder_names_and_paths["hunyuan3d"] = ([hunyuan3d_path], {".safetensors", ".ckpt", ".pt", ".pth"})

    checkpoints = folder_paths.get_filename_list("hunyuan3d")
    if not checkpoints:
        checkpoints = ["model.safetensors"]
    return checkpoints


def apply_mask_to_image(image_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
    """
    Apply mask to image, creating RGBA with transparency where mask is black.

    Args:
        image_np: RGB image array (H, W, 3) float32 [0, 1]
        mask_np: Mask array (H, W) or (H, W, 1) or (B, H, W) float32 [0, 1], white = foreground

    Returns:
        RGBA image array (H, W, 4) uint8 [0, 255]
    """
    print(f"[Hunyuan3D] apply_mask_to_image: image shape={image_np.shape}, mask shape={mask_np.shape}")

    # Handle different mask shapes
    if len(mask_np.shape) == 3:
        if mask_np.shape[0] == 1:
            # (1, H, W) -> (H, W)
            mask_np = mask_np[0]
        elif mask_np.shape[2] == 1:
            # (H, W, 1) -> (H, W)
            mask_np = mask_np[:, :, 0]
        else:
            # (H, W, C) -> take first channel
            mask_np = mask_np[:, :, 0]

    # Ensure image is 3 channels
    if len(image_np.shape) == 3 and image_np.shape[2] == 4:
        # RGBA -> RGB
        image_np = image_np[:, :, :3]

    print(f"[Hunyuan3D] After processing: image shape={image_np.shape}, mask shape={mask_np.shape}")

    # Resize mask to match image if needed
    if mask_np.shape[:2] != image_np.shape[:2]:
        from PIL import Image as PILImage
        mask_pil = PILImage.fromarray((mask_np * 255).astype(np.uint8), 'L')
        mask_pil = mask_pil.resize((image_np.shape[1], image_np.shape[0]), PILImage.Resampling.NEAREST)
        mask_np = np.array(mask_pil).astype(np.float32) / 255.0
        print(f"[Hunyuan3D] Resized mask to {mask_np.shape}")

    # Convert to uint8
    rgb = (image_np * 255).astype(np.uint8)
    alpha = (mask_np * 255).astype(np.uint8)

    # Ensure rgb is (H, W, 3)
    if len(rgb.shape) == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)

    # Ensure alpha is (H, W)
    if len(alpha.shape) == 3:
        alpha = alpha[:, :, 0]

    # Create RGBA
    rgba = np.dstack([rgb, alpha])
    print(f"[Hunyuan3D] Final RGBA shape={rgba.shape}")

    return rgba


def project_colors_to_mesh(mesh, image: Image.Image):
    """
    Project colors from the input image onto mesh vertices.
    Simple front-projection based on vertex positions.

    Args:
        mesh: trimesh object
        image: PIL Image used for generation

    Returns:
        mesh with vertex_colors set
    """
    import trimesh

    if len(mesh.vertices) == 0:
        return mesh

    # Get image as numpy array
    img_np = np.array(image.convert('RGB')).astype(np.float32) / 255.0
    h, w = img_np.shape[:2]

    # Get vertex positions
    vertices = mesh.vertices.copy()

    # Normalize vertices to [0, 1] range based on bounding box
    bounds = mesh.bounds
    if bounds is not None:
        min_b, max_b = bounds[0], bounds[1]
        extent = max_b - min_b
        extent[extent == 0] = 1  # Avoid division by zero

        # Normalize to [0, 1]
        normalized = (vertices - min_b) / extent

        # Map X, Y to image coordinates (front view projection)
        # X -> image width, Y -> image height (inverted)
        u = np.clip(normalized[:, 0], 0, 1)
        v = np.clip(1 - normalized[:, 1], 0, 1)  # Flip Y

        # Convert to pixel coordinates
        px = (u * (w - 1)).astype(int)
        py = (v * (h - 1)).astype(int)

        # Sample colors from image
        vertex_colors = img_np[py, px]

        # Add alpha channel (fully opaque)
        vertex_colors = np.hstack([vertex_colors, np.ones((len(vertex_colors), 1))])

        # Create new mesh with vertex colors
        colored_mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            vertex_colors=(vertex_colors * 255).astype(np.uint8)
        )

        print(f"[Hunyuan3D] Projected colors onto {len(vertices)} vertices")
        return colored_mesh

    return mesh


class LoadHunyuan3DModel:
    """
    Loads the Hunyuan3D-2 model for image-to-3D generation.
    Model is cached for reuse across multiple generations.

    Requirements:
    - diffusers >= 0.31.0
    - transformers
    - accelerate

    Place model files in: ComfyUI/models/hunyuan3d/
    Or use 'tencent/Hunyuan3D-2' to download from HuggingFace.
    """

    CATEGORY = "Hunyuan3D"
    RETURN_TYPES = ("HUNYUAN3D_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = get_hunyuan3d_checkpoints()
        model_options = ["tencent/Hunyuan3D-2", "tencent/Hunyuan3D-2mini"] + checkpoints
        return {
            "required": {
                "model_path": (model_options,),
                "device": (["cuda:0", "cuda:1", "cpu"],),
                "low_vram_mode": ("BOOLEAN", {"default": False}),
            }
        }

    def load(self, model_path: str, device: str, low_vram_mode: bool):
        global _hunyuan3d_model_cache

        cache_key = f"{model_path}_{device}_{low_vram_mode}"

        if cache_key in _hunyuan3d_model_cache:
            print(f"[Hunyuan3D] Using cached model: {model_path} on {device}")
            return (_hunyuan3d_model_cache[cache_key],)

        print(f"[Hunyuan3D] Loading model: {model_path} on {device}...")

        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        except ImportError:
            raise ImportError(
                "Hunyuan3D not installed. Install with:\n"
                "  pip install diffusers>=0.31.0 accelerate transformers\n"
                "  pip install git+https://github.com/Tencent/Hunyuan3D-2.git"
            )

        if model_path.startswith("tencent/"):
            pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
        else:
            import folder_paths
            full_path = folder_paths.get_full_path("hunyuan3d", model_path)
            if full_path is None:
                raise FileNotFoundError(
                    f"Model '{model_path}' not found. "
                    f"Place Hunyuan3D model in: ComfyUI/models/hunyuan3d/"
                )
            model_dir = os.path.dirname(full_path)
            pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_dir)

        if low_vram_mode:
            pipeline.enable_model_cpu_offload()
            print("[Hunyuan3D] Low VRAM mode enabled (CPU offload)")
        else:
            pipeline.to(device)

        _hunyuan3d_model_cache[cache_key] = pipeline
        print(f"[Hunyuan3D] Model loaded successfully on {device}")

        return (pipeline,)


class ImageTo3DMeshHunyuan:
    """
    Converts an input image to a 3D mesh using Hunyuan3D-2.

    Inputs:
    - image: The source image
    - mask (optional): White = foreground, Black = background.
      If provided, creates RGBA image with transparency.

    The output mesh is compatible with RenderMesh8Directions node.
    """

    CATEGORY = "Hunyuan3D"
    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HUNYUAN3D_MODEL",),
                "image": ("IMAGE",),
                "num_inference_steps": ("INT", {"default": 30, "min": 10, "max": 100, "step": 5}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 15.0, "step": 0.5}),
                "octree_resolution": ("INT", {"default": 256, "min": 128, "max": 512, "step": 64}),
                "project_colors": ("BOOLEAN", {"default": True}),
                "unload_model": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "mask": ("MASK",),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffff}),
            }
        }

    def generate(
        self,
        model,
        image: torch.Tensor,
        num_inference_steps: int,
        guidance_scale: float,
        octree_resolution: int,
        project_colors: bool,
        unload_model: bool,
        mask: torch.Tensor = None,
        seed: int = -1
    ):
        global _hunyuan3d_model_cache

        # Free up VRAM
        try:
            import comfy.model_management as mm
            print("[Hunyuan3D] Unloading ComfyUI models to free VRAM...")
            mm.unload_all_models()
            mm.soft_empty_cache()
        except Exception as e:
            print(f"[Hunyuan3D] Could not unload ComfyUI models: {e}")

        # Convert image to numpy
        img_np = image[0].cpu().numpy()  # (H, W, C) float32 [0, 1]

        # If mask provided, apply it to create RGBA
        if mask is not None:
            mask_np = mask.cpu().numpy()
            print(f"[Hunyuan3D] Raw mask shape: {mask_np.shape}")

            # Handle ComfyUI mask format (can be various shapes)
            if len(mask_np.shape) == 4:
                # (B, C, H, W) or (B, H, W, C)
                mask_np = mask_np[0]
            if len(mask_np.shape) == 3:
                if mask_np.shape[0] == 1:
                    mask_np = mask_np[0]  # (1, H, W) -> (H, W)
                elif mask_np.shape[2] == 1:
                    mask_np = mask_np[:, :, 0]  # (H, W, 1) -> (H, W)

            print(f"[Hunyuan3D] Applying mask to image, mask shape: {mask_np.shape}")
            rgba_np = apply_mask_to_image(img_np, mask_np)
            pil_image = Image.fromarray(rgba_np, 'RGBA')
            print(f"[Hunyuan3D] Input: RGBA image with mask {pil_image.size}")
        elif img_np.shape[2] == 4:
            # Already RGBA
            pil_image = Image.fromarray((img_np * 255).astype(np.uint8), 'RGBA')
            print(f"[Hunyuan3D] Input: RGBA image {pil_image.size}")
        else:
            # RGB without mask - warn user
            pil_image = Image.fromarray((img_np * 255).astype(np.uint8), 'RGB')
            print(f"[Hunyuan3D] Input: RGB image {pil_image.size}")
            print("[Hunyuan3D] WARNING: No mask provided! Background will be included in 3D model.")
            print("[Hunyuan3D] Connect a MASK input to separate foreground from background.")

        # Store original for color projection
        original_image = pil_image.copy()

        # Preprocess: resize foreground and convert to RGB
        if pil_image.mode == 'RGBA':
            pil_image = resize_foreground(pil_image, ratio=0.85)
            pil_image = rgba_to_rgb_gray_background(pil_image)

        # Set seed
        generator = None
        if seed >= 0:
            generator = torch.Generator().manual_seed(seed)
            print(f"[Hunyuan3D] Using seed: {seed}")

        print(f"[Hunyuan3D] Generating 3D mesh (steps={num_inference_steps}, guidance={guidance_scale})...")

        with torch.no_grad():
            result = model(
                image=pil_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                octree_resolution=octree_resolution,
                generator=generator
            )

        mesh = result[0]

        print(f"[Hunyuan3D] Mesh generated: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        if len(mesh.vertices) > 0:
            verts = mesh.vertices
            print(f"[Hunyuan3D] Mesh bounds: X=[{verts[:,0].min():.3f}, {verts[:,0].max():.3f}], "
                  f"Y=[{verts[:,1].min():.3f}, {verts[:,1].max():.3f}], "
                  f"Z=[{verts[:,2].min():.3f}, {verts[:,2].max():.3f}]")

            # Project colors from original image
            if project_colors:
                mesh = project_colors_to_mesh(mesh, original_image)

        if unload_model:
            print("[Hunyuan3D] Unloading model to free VRAM...")
            model.to("cpu")
            _hunyuan3d_model_cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[Hunyuan3D] Model unloaded, VRAM freed")

        return (mesh,)


class LoadHunyuan3DTextureModel:
    """
    Loads the Hunyuan3D-2 texture/paint model for adding textures to meshes.
    """

    CATEGORY = "Hunyuan3D"
    RETURN_TYPES = ("HUNYUAN3D_TEXTURE_MODEL",)
    RETURN_NAMES = ("texture_model",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (["tencent/Hunyuan3D-2"],),
                "device": (["cuda:0", "cuda:1", "cpu"],),
                "low_vram_mode": ("BOOLEAN", {"default": True}),
            }
        }

    def load(self, model_path: str, device: str, low_vram_mode: bool):
        global _hunyuan3d_texture_model_cache

        cache_key = f"tex_{model_path}_{device}_{low_vram_mode}"

        if cache_key in _hunyuan3d_texture_model_cache:
            print(f"[Hunyuan3D-Tex] Using cached texture model")
            return (_hunyuan3d_texture_model_cache[cache_key],)

        print(f"[Hunyuan3D-Tex] Loading texture model: {model_path} on {device}...")

        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
        except ImportError:
            raise ImportError(
                "Hunyuan3D texture module not installed. Install with:\n"
                "  pip install git+https://github.com/Tencent/Hunyuan3D-2.git"
            )

        pipeline = Hunyuan3DPaintPipeline.from_pretrained(model_path)

        if low_vram_mode:
            pipeline.enable_model_cpu_offload()
            print("[Hunyuan3D-Tex] Low VRAM mode enabled")
        else:
            pipeline.to(device)

        _hunyuan3d_texture_model_cache[cache_key] = pipeline
        print(f"[Hunyuan3D-Tex] Texture model loaded successfully")

        return (pipeline,)


class TextureMeshHunyuan:
    """
    Applies textures to a mesh using Hunyuan3D-2 Paint pipeline.

    Takes a mesh and reference image, generates UV-mapped textures.
    """

    CATEGORY = "Hunyuan3D"
    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("textured_mesh",)
    FUNCTION = "texture"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "texture_model": ("HUNYUAN3D_TEXTURE_MODEL",),
                "mesh": ("MESH",),
                "image": ("IMAGE",),
                "unload_model": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    def texture(
        self,
        texture_model,
        mesh,
        image: torch.Tensor,
        unload_model: bool,
        mask: torch.Tensor = None
    ):
        global _hunyuan3d_texture_model_cache

        # Free up VRAM
        try:
            import comfy.model_management as mm
            print("[Hunyuan3D-Tex] Unloading ComfyUI models to free VRAM...")
            mm.unload_all_models()
            mm.soft_empty_cache()
        except Exception as e:
            print(f"[Hunyuan3D-Tex] Could not unload ComfyUI models: {e}")

        # Convert image to PIL
        img_np = image[0].cpu().numpy()

        # If mask provided, apply it
        if mask is not None:
            mask_np = mask.cpu().numpy()
            if len(mask_np.shape) == 3:
                mask_np = mask_np[0]
            rgba_np = apply_mask_to_image(img_np, mask_np)
            pil_image = Image.fromarray(rgba_np, 'RGBA')
            print(f"[Hunyuan3D-Tex] Input: RGBA image with mask {pil_image.size}")
        elif img_np.shape[2] == 4:
            pil_image = Image.fromarray((img_np * 255).astype(np.uint8), 'RGBA')
            print(f"[Hunyuan3D-Tex] Input: RGBA image {pil_image.size}")
        else:
            pil_image = Image.fromarray((img_np * 255).astype(np.uint8), 'RGB')
            print(f"[Hunyuan3D-Tex] Input: RGB image {pil_image.size}")

        print(f"[Hunyuan3D-Tex] Generating textures for mesh with {len(mesh.vertices)} vertices...")

        with torch.no_grad():
            textured_mesh = texture_model(mesh, image=pil_image)

        print(f"[Hunyuan3D-Tex] Texturing complete")

        if unload_model:
            print("[Hunyuan3D-Tex] Unloading texture model...")
            texture_model.to("cpu")
            _hunyuan3d_texture_model_cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[Hunyuan3D-Tex] Model unloaded")

        return (textured_mesh,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "LoadHunyuan3DModel": LoadHunyuan3DModel,
    "ImageTo3DMeshHunyuan": ImageTo3DMeshHunyuan,
    "LoadHunyuan3DTextureModel": LoadHunyuan3DTextureModel,
    "TextureMeshHunyuan": TextureMeshHunyuan,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadHunyuan3DModel": "Load Hunyuan3D Model",
    "ImageTo3DMeshHunyuan": "Image to 3D Mesh (Hunyuan3D)",
    "LoadHunyuan3DTextureModel": "Load Hunyuan3D Texture Model",
    "TextureMeshHunyuan": "Texture Mesh (Hunyuan3D)",
}
