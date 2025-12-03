import os
from PIL import Image
import torch
import numpy as np


def get_first_image(img):
    # 递归取到最内层的非list对象，遇到空list返回None
    while isinstance(img, list):
        if not img:  # 空list
            return None
        img = img[0]
    return img

class SaveImageWithName:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename": ("STRING",),
                "filename_suffix": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "save_path": ("STRING", {
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
            pil_image = Image.fromarray(image)
            
            # 处理EXIF方向信息，防止图片旋转
            try:
                # 获取EXIF信息
                exif = pil_image.getexif()
                if exif:
                    # 获取方向信息
                    orientation = exif.get(274)  # 274是方向标签的ID
                    if orientation:
                        # 根据方向信息旋转图片
                        rotation_values = {
                            3: 180,
                            6: 270,
                            8: 90
                        }
                        if orientation in rotation_values:
                            pil_image = pil_image.rotate(rotation_values[orientation], expand=True)
                        # 清除EXIF方向信息，防止重复旋转
                        exif[274] = 1  # 设置为正常方向
                        pil_image.info['exif'] = exif.tobytes()
            except Exception as e:
                print(f"[DaShuai] EXIF处理警告: {str(e)}")
            
            return pil_image
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

    def process_filename(self, filename, suffix):
        """处理文件名，将后缀拼接到合适的位置"""
        if not suffix:
            return filename
        
        # 检查文件名中是否有'.'符号
        if '.' in filename:
            # 有'.'符号，拼接到'.'前面
            name, ext = os.path.splitext(filename)
            return f"{name}{suffix}{ext}"
        else:
            # 没有'.'符号，直接拼接到文件名后面
            return f"{filename}{suffix}"

    def save_image(self, image, filename, save_path, filename_suffix):
        try:
            os.makedirs(save_path, exist_ok=True)

            # 强制转换为list处理
            if isinstance(image, list):
                if isinstance(filename, str):
                    # 自动扩展文件名
                    name, ext = os.path.splitext(filename)
                    ext = ext if ext else ".png"
                    filename = [f"{name}（{i+1}）{ext}" for i in range(len(image))]
                
                # 统一list对list处理
                for img, name in zip(image, filename):
                    img = get_first_image(img)
                    if img is None:
                        continue
                    if not isinstance(name, str):
                        name = str(name)
                    if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        name += ".png"
                    
                    # 处理拼接文件名
                    processed_name = self.process_filename(name, filename_suffix)
                    
                    # 检查重名并生成唯一文件名
                    unique_name = self.get_unique_filename(save_path, processed_name)
                    full_path = os.path.join(save_path, unique_name)
                    
                    img_pil = self.tensor_to_pil(img)
                    img_pil.save(full_path)
                    print(f"[DaShuai] 图片已保存: {full_path}")
            else:
                # 单张图片处理
                img = get_first_image(image)
                if img is None:
                    print("[DaShuai] 跳过空图片")
                    return ()
                if not isinstance(filename, str):
                    filename = str(filename)
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    filename += '.png'
                
                # 处理拼接文件名
                processed_name = self.process_filename(filename, filename_suffix)
                
                # 检查重名并生成唯一文件名
                unique_name = self.get_unique_filename(save_path, processed_name)
                full_path = os.path.join(save_path, unique_name)
                
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
    "SaveImageWithName": "SaveImageWithName☀"
}
