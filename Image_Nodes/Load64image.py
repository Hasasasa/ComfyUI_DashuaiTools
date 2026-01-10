import base64
import io

import numpy as np
import torch
from PIL import Image


class Load64image:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base64_image": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Paste base64 here (optionally with data:image/...;base64, prefix)",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load"
    CATEGORY = "Image/Load"
    OUTPUT_NODE = True

    def load(self, base64_image: str):
        if not isinstance(base64_image, str) or not base64_image.strip():
            raise ValueError("base64_image is empty")

        data = base64_image.strip()
        if "," in data and "base64" in data.split(",", 1)[0].lower():
            data = data.split(",", 1)[1]

        data = "".join(data.split())
        try:
            raw = base64.b64decode(data, validate=True)
        except Exception:
            raw = base64.b64decode(data)

        try:
            img = Image.open(io.BytesIO(raw))
            img = img.convert("RGB")
        except Exception as e:
            raise ValueError(f"invalid image data: {e}") from e

        img_tensor = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_tensor)[None,]
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
        return (img_tensor,)


NODE_CLASS_MAPPINGS = {
    "Load64image": Load64image,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load64image": "Load64image☀",
}

