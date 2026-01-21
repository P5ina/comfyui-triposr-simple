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

    For best results, provide an RGBA image with transparent background.
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
                "unload_model": ("BOOLEAN", {"default": True}),
            },
            "optional": {
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
        unload_model: bool,
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

        # Convert to PIL
        img_np = image[0].cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        if img_np.shape[2] == 4:
            pil_image = Image.fromarray(img_np, 'RGBA')
            print(f"[Hunyuan3D] Input: RGBA image {pil_image.size}")

            pil_image = resize_foreground(pil_image, ratio=0.85)
            pil_image = rgba_to_rgb_gray_background(pil_image)
        else:
            pil_image = Image.fromarray(img_np, 'RGB')
            print(f"[Hunyuan3D] Input: RGB image {pil_image.size}")
            print("[Hunyuan3D] Warning: For best results, use RGBA image with transparent background")

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

        if unload_model:
            print("[Hunyuan3D] Unloading model to free VRAM...")
            model.to("cpu")
            _hunyuan3d_model_cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[Hunyuan3D] Model unloaded, VRAM freed")

        return (mesh,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "LoadHunyuan3DModel": LoadHunyuan3DModel,
    "ImageTo3DMeshHunyuan": ImageTo3DMeshHunyuan,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadHunyuan3DModel": "Load Hunyuan3D Model",
    "ImageTo3DMeshHunyuan": "Image to 3D Mesh (Hunyuan3D)",
}
