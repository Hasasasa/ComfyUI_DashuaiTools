import numpy as np
import torch
from skimage import filters
from skimage.morphology import square
#转载请保留该标签。V：sundashuaio
class 最小值滤镜:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "图像": ("IMAGE",),
                "半径": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 255,
                    "step": 1,
                    "display": "number"
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_minimum_filter"
    CATEGORY = "Image/Filter"

    def apply_minimum_filter(self, 图像, 半径):
        if isinstance(图像, torch.Tensor):
            图像 = 图像.cpu().numpy()
        #转载请保留该标签。V：sundashuaio
        # 处理 4 维张量
        if 图像.ndim == 4:
            filtered_images = []
            for img in 图像:
                image_uint8 = (img * 255).astype(np.uint8)

                if image_uint8.ndim == 3 and image_uint8.shape[2] == 3:
                    filtered = np.stack([
                        filters.rank.minimum(image_uint8[..., i], square(半径))
                        for i in range(3)
                    ], axis=-1)
                else:
                    filtered = filters.rank.minimum(image_uint8, square(半径))

                filtered = filtered.astype(np.float32) / 255.0
                filtered_images.append(filtered)
            filtered = np.stack(filtered_images)
        else:
            image_uint8 = (图像 * 255).astype(np.uint8)

            if image_uint8.ndim == 3 and image_uint8.shape[2] == 3:
                filtered = np.stack([
                    filters.rank.minimum(image_uint8[..., i], square(半径))
                    for i in range(3)
                ], axis=-1)
            else:
                filtered = filters.rank.minimum(image_uint8, square(半径))

            filtered = filtered.astype(np.float32) / 255.0

        return (torch.from_numpy(filtered),)

NODE_CLASS_MAPPINGS = {
    "D_SHINENode": 最小值滤镜
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "D_SHINENode": "最小值滤镜 ☀"
}