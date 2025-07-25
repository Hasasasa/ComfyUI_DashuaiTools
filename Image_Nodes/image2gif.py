import os
from PIL import Image
import numpy as np
import torch
import io
import base64
from tqdm import tqdm  # 添加 tqdm 导入

class ImageComparisonGIF:
    def __init__(self):
        self.max_image_size = 1024

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "frames": ("INT", {"default": 20, "min": 2, "max": 100}),
                "output_path": ("STRING", {
                    "default": "compare.gif",
                    "multiline": False
                }),
            },
        }

    RETURN_TYPES = ("STRING",)  # 移除 GIF 返回类型
    RETURN_NAMES = ("gif_path",)  # 只返回文件路径
    FUNCTION = "create_gif"
    CATEGORY = "Image/GIF"
    OUTPUT_NODE = True

    @classmethod
    def PREVIEW_TYPE(s):
        return ["gif"]

    def tensor_to_pil(self, image_tensor):
        if isinstance(image_tensor, torch.Tensor):
            image = image_tensor.cpu().numpy()

        # 如果是4维，比如 batch 形式 [B, C, H, W]，取第一个
        if image.ndim == 4:
            image = image[0]

        # C, H, W -> H, W, C
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))

        image_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(image_uint8, mode="RGB")

    def execute(self, image1, image2, frames, output_path):
        try:
            if hasattr(self, 'progress_bar') and callable(self.progress_bar):
                result = self.create_gif(image1, image2, frames, output_path)
                return result
            else:
                print("[DaShuai] 警告: progress_bar 未设置或不可调用")
                result = self.create_gif(image1, image2, frames, output_path)
                return result
        except Exception as e:
            print(f"[DaShuai] 执行失败: {str(e)}")
            raise e

    @classmethod
    def get_unique_filename(self, output_path):
        """获取唯一的文件名，如果文件存在则添加数字后缀"""
        name, ext = os.path.splitext(output_path)
        counter = 1
        new_path = output_path
        
        while os.path.exists(new_path):
            new_path = f"{name}_{counter}{ext}"
            counter += 1
        
        return new_path

    def create_gif(self, image1, image2, frames, output_path):
        try:
            # 修改输出路径构建方式
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "output")
            base_output_path = os.path.join(output_dir, output_path)
            
            # 获取唯一文件名
            full_output_path = self.get_unique_filename(base_output_path)
            
            print(f"[DaShuai] 输出路径: {full_output_path}")
            
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 转换和验证图像
            img1 = self.tensor_to_pil(image1)
            img2 = self.tensor_to_pil(image2)
            
            # 调整图像大小（如果需要）
            if max(img1.size) > self.max_image_size:
                ratio = self.max_image_size / max(img1.size)
                new_size = tuple(int(dim * ratio) for dim in img1.size)
                img1 = img1.resize(new_size, Image.LANCZOS)
                img2 = img2.resize(new_size, Image.LANCZOS)
            else:
                img2 = img2.resize(img1.size, Image.LANCZOS)

            # 创建GIF帧
            width, height = img1.size
            gif_frames = []
            
            for i in range(frames):
                mask_x = int(width * (i / (frames - 1)))
                new_frame = Image.new("RGB", (width, height))
                new_frame.paste(img1.crop((0, 0, mask_x, height)), (0, 0))
                new_frame.paste(img2.crop((mask_x, 0, width, height)), (mask_x, 0))
                gif_frames.append(new_frame)

            # 保存GIF
            print("[DaShuai] 正在保存GIF...")
            gif_frames[0].save(
                full_output_path,
                save_all=True,
                append_images=gif_frames[1:],
                duration=100,
                loop=0,
                optimize=True
            )

            print(f"[DaShuai] 创建GIF成功: {full_output_path}")
            return (full_output_path,)  # 只返回文件路径
            
        except Exception as e:
            print(f"[DaShuai] 创建GIF失败: {str(e)}")
            raise e

# 节点映射
NODE_CLASS_MAPPINGS = {
    "ImageComparisonGIF": ImageComparisonGIF
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageComparisonGIF": "图像滑动对比GIF ☀"
}