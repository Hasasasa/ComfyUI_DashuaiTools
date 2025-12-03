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
                "folder_path": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "max_output": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
                "start_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "step": 1
                }),
                "sort_mode": (["无", "Alphabetical (ASC)", "Alphabetical (DESC)", 
                               "Numerical (ASC)", "Numerical (DESC)",
                               "Datetime (ASC)", "Datetime (DESC)"],),
                "always_reload": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "file_name")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "load_images"
    CATEGORY = "Image/Load"
    OUTPUT_NODE = True

    def load_images(self, folder_path, max_output=0, start_index=0, sort_mode="无", always_reload=False):
        try:
            # 确保文件夹路径存在
            if not os.path.exists(folder_path):
                raise ValueError(f"文件夹不存在: {folder_path}")
                
            # 获取所有图片文件
            image_files = []
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp'):
                image_files.extend(glob.glob(os.path.join(folder_path, ext)))
                
            # 根据排序方法对文件列表进行排序
            if sort_mode != "无":
                if "Alphabetical" in sort_mode:
                    image_files.sort(reverse="DESC" in sort_mode)
                elif "Numerical" in sort_mode:
                    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))), 
                                   reverse="DESC" in sort_mode)
                elif "Datetime" in sort_mode:
                    image_files.sort(key=os.path.getmtime, reverse="DESC" in sort_mode)
            
            # 应用起始索引和最大图片数限制
            if max_output == 0:
                image_files = image_files[start_index:]
            else:
                image_files = image_files[start_index:start_index + max_output]
            
            if not image_files:
                raise ValueError(f"在文件夹中未找到图片: {folder_path}")
                
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
    "LoadImageList": "LoadImageList☀"
}
