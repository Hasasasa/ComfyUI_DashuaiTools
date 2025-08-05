import os
import glob
from PIL import Image
import numpy as np
import torch

class LoadImageList:
    def __init__(self):
        self.image_list = []
        self.current_index = 0
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "文件夹路径": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "最大输出数": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
                "起始加载数": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "step": 1
                }),
                "排序模式": (["无", "Alphabetical (ASC)", "Alphabetical (DESC)", 
                               "Numerical (ASC)", "Numerical (DESC)",
                               "Datetime (ASC)", "Datetime (DESC)"],),
                "始终加载": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "file_name")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "load_images"
    CATEGORY = "Image/Load"
    OUTPUT_NODE = True

    def load_images(self, 文件夹路径, 最大输出数=0, 起始加载数=0, 排序模式="无", 始终加载=False):
        try:
            # 确保文件夹路径存在
            if not os.path.exists(文件夹路径):
                raise ValueError(f"文件夹不存在: {文件夹路径}")
                
            # 获取所有图片文件
            image_files = []
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp'):
                image_files.extend(glob.glob(os.path.join(文件夹路径, ext)))
                
            # 根据排序方法对文件列表进行排序
            if 排序模式 != "无":
                if "Alphabetical" in 排序模式:
                    image_files.sort(reverse="DESC" in 排序模式)
                elif "Numerical" in 排序模式:
                    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))), 
                                   reverse="DESC" in 排序模式)
                elif "Datetime" in 排序模式:
                    image_files.sort(key=os.path.getmtime, reverse="DESC" in 排序模式)
            
            # 应用起始索引和最大图片数限制
            if 最大输出数 == 0:
                image_files = image_files[起始加载数:]
            else:
                image_files = image_files[起始加载数:起始加载数 + 最大输出数]
            
            if not image_files:
                raise ValueError(f"在文件夹中未找到图片: {文件夹路径}")
                
            # 加载图片
            loaded_images = []
            file_names = []
            
            for img_path in image_files:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = np.array(img).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_tensor)[None,]
                    if len(img_tensor.shape) == 3:
                        img_tensor = img_tensor.unsqueeze(0)
                    loaded_images.append(img_tensor)
                    file_names.append(os.path.basename(img_path))  # 只保留文件名，不包含路径
                except Exception as e:
                    print(f"[DaShuai] 加载图片失败 {img_path}: {str(e)}")
                    continue
            
            if not loaded_images:
                raise ValueError("没有成功加载任何图片")
            
            print(f"[DaShuai] 成功加载 {len(loaded_images)} 张图片")
            return (loaded_images, file_names)
            
        except Exception as e:
            print(f"[DaShuai] 加载图片失败: {str(e)}")
            raise e

# 节点映射
NODE_CLASS_MAPPINGS = {
    "LoadImageList": LoadImageList
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageList": "批量加载图像 ☀"
}