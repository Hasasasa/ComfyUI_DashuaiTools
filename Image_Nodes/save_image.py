import os
from PIL import Image
import torch
import numpy as np

class SaveImageWithName:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "图片": ("IMAGE",),
                "文件名": ("STRING",),
                "保存路径": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
            }
        }
    
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save_image"
    CATEGORY = "Image/Save"
    OUTPUT_NODE = True

    def tensor_to_pil(self, image_tensor):
        if isinstance(image_tensor, torch.Tensor):
            # 确保张量在CPU上
            image = image_tensor.cpu().numpy()
            # 如果是批处理格式，取第一张图片
            if len(image.shape) == 4:
                image = image[0]
            # 调整通道顺序 
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            # 转换为uint8格式
            image = (image * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(image)
        return image_tensor

    def save_image(self, 图片, 文件名, 保存路径):
        try:
            os.makedirs(保存路径, exist_ok=True)
            # 如果图片是 torch.Tensor 且文件名是 list
            if isinstance(图片, torch.Tensor) and isinstance(文件名, list):
                for i in range(min(图片.shape[0], len(文件名))):
                    name = 文件名[i]
                    if not isinstance(name, str):
                        name = str(name)
                    if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        name += ".png"
                    full_path = os.path.join(保存路径, name)
                    img_pil = self.tensor_to_pil(图片[i])
                    img_pil.save(full_path)
                    print(f"[DaShuai] 图片已保存: {full_path}")
            # 如果图片和文件名都是 list
            elif isinstance(图片, list) and isinstance(文件名, list):
                for img, name in zip(图片, 文件名):
                    if not isinstance(name, str):
                        name = str(name)
                    if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        name += ".png"
                    full_path = os.path.join(保存路径, name)
                    img_pil = self.tensor_to_pil(img)
                    img_pil.save(full_path)
                    print(f"[DaShuai] 图片已保存: {full_path}")
            # 单张图片和单个文件名
            else:
                if not isinstance(文件名, str):
                    文件名 = str(文件名)
                if not 文件名.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    文件名 += ".png"
                full_path = os.path.join(保存路径, 文件名)
                img = self.tensor_to_pil(图片)
                img.save(full_path)
                print(f"[DaShuai] 图片已保存: {full_path}")
            return ()
        except Exception as e:
            print(f"[DaShuai] 保存图片失败: {str(e)}")
            raise e

# 节点映射
NODE_CLASS_MAPPINGS = {
    "SaveImageWithName": SaveImageWithName
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageWithName": "保存图像(自定义名称) ☀"
}