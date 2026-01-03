import base64
import io
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch


def _get_first_image(image):
    while isinstance(image, list):
        if not image:
            return None
        image = image[0]
    return image


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be torch.Tensor")
    arr = image.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    return Image.fromarray(arr).convert("RGB")


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be PIL.Image")
    rgb = image.convert("RGB")
    arr = np.array(rgb).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None,]


class Gemini_API_Image:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "API_Key": ("STRING", {"default": "<your_key>"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "rows": 4}),
                "model_name": ("STRING", {"default": "gemini-3-pro-image-preview"}),
                "aspect_ratio": (["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"], {"default": "5:4"}),
                "resolution": (["1K", "2K", "4K"], {"default": "2K"}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            },
            "optional": {
                "aspect_ratio_text": ("STRING", {"default": "", "forceInput": True}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "image_9": ("IMAGE",),
                "image_10": ("IMAGE",),
                "image_11": ("IMAGE",),
                "image_12": ("IMAGE",),
                "image_13": ("IMAGE",),
                "image_14": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "text")
    FUNCTION = "generate"
    CATEGORY = "DaNodes/API"

    def _pil_to_png_bytes(self, image: Image.Image) -> bytes:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    def _build_contents(self, prompt: str, images: List[Image.Image], types_mod) -> list:
        contents = []
        if prompt:
            contents.append(types_mod.Part.from_text(text=prompt))
        for img in images:
            data = self._pil_to_png_bytes(img)
            contents.append(types_mod.Part.from_bytes(data=data, mime_type="image/png"))
        return contents

    def _import_genai(self):
        try:
            from google import genai as genai_mod
            from google.genai import types as types_mod
            return genai_mod, types_mod, None
        except Exception:
            pass
        try:
            import importlib

            genai_mod = importlib.import_module("google.genai")
            types_mod = importlib.import_module("google.genai.types")
            return genai_mod, types_mod, None
        except Exception as e:
            return None, None, e

    def _collect_reference_images(self, *images) -> List[Image.Image]:
        refs: List[Image.Image] = []
        for entry in images:
            if entry is None:
                continue
            entry = _get_first_image(entry)
            if entry is None:
                continue
            if isinstance(entry, Image.Image):
                refs.append(entry.convert("RGB"))
                continue
            if isinstance(entry, torch.Tensor):
                refs.append(_tensor_to_pil(entry))
                continue
            raise TypeError("image inputs must be IMAGE tensors")
        return refs[:14]

    def _extract_response_parts(self, response) -> Tuple[Optional[Image.Image], str]:
        parts = getattr(response, "parts", None)
        if parts is None and hasattr(response, "candidates"):
            try:
                parts = response.candidates[0].content.parts
            except Exception:
                parts = []
        if parts is None:
            parts = []

        text_parts: List[str] = []
        output_image: Optional[Image.Image] = None

        response_text = getattr(response, "text", None)
        if isinstance(response_text, str) and response_text.strip():
            text_parts.append(response_text.strip())

        for part in parts:
            text = getattr(part, "text", None)
            if text:
                text_parts.append(text)
                continue
            inline = getattr(part, "inline_data", None)
            data = getattr(inline, "data", None) if inline is not None else None
            if data:
                try:
                    raw = base64.b64decode(data) if isinstance(data, str) else data
                    output_image = Image.open(io.BytesIO(raw)).convert("RGB")
                except Exception:
                    output_image = None
                continue
            try:
                image = part.as_image()
            except Exception:
                image = None
            if image is None:
                continue
            if isinstance(image, Image.Image):
                output_image = image.convert("RGB")
            elif isinstance(image, (bytes, bytearray)):
                try:
                    output_image = Image.open(io.BytesIO(image)).convert("RGB")
                except Exception:
                    output_image = None
            else:
                try:
                    output_image = Image.open(image).convert("RGB")
                except Exception:
                    output_image = None
        return output_image, "\n".join(t for t in text_parts if t).strip()

    def _empty_image(self) -> torch.Tensor:
        return torch.zeros((1, 1, 1, 3), dtype=torch.float32)

    def generate(
        self,
        API_Key: str,
        prompt: str,
        model_name: str,
        aspect_ratio: str,
        resolution: str,
        noise_seed: int = 0,
        aspect_ratio_text: str = "",
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
        image_9=None,
        image_10=None,
        image_11=None,
        image_12=None,
        image_13=None,
        image_14=None,
    ) -> Tuple[torch.Tensor, str]:
        if not API_Key or API_Key.strip() in ("<your_key>", "your_key", "api_key"):
            return (self._empty_image(), "API key is missing.")

        genai, types, import_err = self._import_genai()
        if import_err is not None:
            msg = (
                "google-genai is not available or the 'google' package is shadowing it. "
                f"Original error: {import_err}. "
                "Try: pip uninstall -y google; pip install -U google-genai"
            )
            return (self._empty_image(), msg)

        ref_images = self._collect_reference_images(
            image_1,
            image_2,
            image_3,
            image_4,
            image_5,
            image_6,
            image_7,
            image_8,
            image_9,
            image_10,
            image_11,
            image_12,
            image_13,
            image_14,
        )

        if not prompt and not ref_images:
            return (self._empty_image(), "Prompt and reference images are both empty.")

        contents = self._build_contents(prompt, ref_images, types)

        try:
            client = genai.Client(api_key=API_Key)
            aspect_ratio_value = aspect_ratio_text.strip() or aspect_ratio
            image_cfg = types.ImageConfig(
                aspect_ratio=aspect_ratio_value,
                image_size=resolution,
            )
            if noise_seed:
                try:
                    setattr(image_cfg, "seed", int(noise_seed))
                except Exception:
                    pass
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=image_cfg,
                ),
            )
        except Exception as e:
            return (self._empty_image(), f"Gemini request failed: {e}")

        output_image, text = self._extract_response_parts(response)
        if output_image is None:
            return (self._empty_image(), text or "No image returned from Gemini.")

        return (_pil_to_tensor(output_image), text)


NODE_CLASS_MAPPINGS = {
    "Gemini_API_Image": Gemini_API_Image,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemini_API_Image": "Gemini_API_Image☀",
}
