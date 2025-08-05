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

    def get_unique_filename(self, base_path, filename):
        """生成唯一的文件名，避免覆盖"""
        name, ext = os.path.splitext(filename)
        counter = 1
        new_filename = filename
        
        while os.path.exists(os.path.join(base_path, new_filename)):
            new_filename = f"{name}_{counter}{ext}"
            counter += 1
        
        return new_filename

    def save_image(self, 图片, 文件名, 保存路径):
        try:
            os.makedirs(保存路径, exist_ok=True)

            # 强制转换为list处理
            if isinstance(图片, list):
                if isinstance(文件名, str):
                    # 自动扩展文件名
                    name, ext = os.path.splitext(文件名)
                    ext = ext if ext else ".png"
                    文件名 = [f"{name}（{i+1}）{ext}" for i in range(len(图片))]
                
                # 统一list对list处理
                for img, name in zip(图片, 文件名):
                    img = get_first_image(img)
                    if img is None:
                        continue
                    if not isinstance(name, str):
                        name = str(name)
                    if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        name += ".png"
                    
                    # 检查重名并生成唯一文件名
                    unique_name = self.get_unique_filename(保存路径, name)
                    full_path = os.path.join(保存路径, unique_name)
                    
                    img_pil = self.tensor_to_pil(img)
                    img_pil.save(full_path)
                    print(f"[DaShuai] 图片已保存: {full_path}")
            else:
                # 单张图片处理
                img = get_first_image(图片)
                if img is None:
                    print("[DaShuai] 跳过空图片")
                    return ()
                if not isinstance(文件名, str):
                    文件名 = str(文件名)
                if not 文件名.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    文件名 += '.png'
                
                # 检查重名并生成唯一文件名
                unique_name = self.get_unique_filename(保存路径, 文件名)
                full_path = os.path.join(保存路径, unique_name)
                
                img_pil = self.tensor_to_pil(img)
                img_pil.save(full_path)
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

def get_first_image(img):
    # 递归取到最内层的非list对象，遇到空list返回None
    while isinstance(img, list):
        if not img:  # 空list
            return None
        img = img[0]
    return img