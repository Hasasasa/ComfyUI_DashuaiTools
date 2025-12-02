import json
import copy
import traceback
import re
from typing import Any, Dict, List
import sys
import os
import uuid
import hashlib

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- 核心依赖导入 ---
try:
    import nodes as comfy_nodes
    import execution
except ImportError:
    # 尝试自动查找 ComfyUI 路径
    try:
        current_path = os.path.dirname(os.path.abspath(__file__))
        # 假设位于 custom_nodes/ComfyUI_DashuaiTools/Image_Nodes/XY_Image.py
        # 回溯 3 层找到 ComfyUI 根目录
        comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_path))))
        if comfy_path not in sys.path:
            sys.path.insert(0, comfy_path)
        import nodes as comfy_nodes
        import execution
    except Exception as e:
        print(f"[XY_Image] CRITICAL WARNING: Could not import 'nodes' or 'execution'. {e}")
        comfy_nodes = None
        execution = None

# --- 全局缓存: 用于存储上一次生成的图片列表 ---
# 结构: { "content_hash_string": [tensor1, tensor2, ...] }
# 为了节省显存，我们永远只保留最后一次成功的生成结果
_XY_GLOBAL_CACHE = {}

class XYImage:
    """
    独立实现的 XY Plot 节点。
    能够解析 X/Y 轴配置，修改 Prompt 并重新执行工作流生成网格图。
    支持缓存机制：仅修改 Grid/Font 样式时不会重新跑图。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                # 修改键名以在前端显示中文
                "间隔大小": ("INT", {"default": 10, "min": 5, "max": 1000, "step": 5}),
                "字体大小": ("INT", {"default": 50, "min": 50, "max": 250, "step": 1}),
                "展示面板": (["XY Plot", "X:Y", "Y:X"],),
                "x_attr": ("STRING", {"default": "", "multiline": True, "placeholder": "X 轴配置: <x:Label>[node_id:input]='value'"}),
                "y_attr": ("STRING", {"default": "", "multiline": True, "placeholder": "Y 轴配置: <y:Label>[node_id:input]='value'"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "my_unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "xyimage")
    FUNCTION = "build_grid"
    CATEGORY = "Image/XYZ"

    @classmethod
    def _calculate_content_hash(cls, x_attr, y_attr, prompt, ignore_id=None):
        """
        计算“内容生成”相关的哈希值。
        策略升级：计算【整个工作流 + XY配置】的指纹。
        这样只要工作流里任何参数变了（比如改了提示词），哈希就会变，从而触发更新。
        **关键修复**：通过 ignore_id 排除当前节点自己，防止修改字体/间隔导致哈希变化。
        """
        # 1. 基础内容 (XY配置)
        # x_attr 和 y_attr 是必须要算的，因为它们定义了绘图逻辑
        content_str = f"{x_attr}_{y_attr}"
        
        # 2. 全图指纹 (Workflow Fingerprint)
        if prompt:
            try:
                # 排序节点 ID 以保证顺序稳定
                node_ids = sorted(list(prompt.keys()))
                for nid in node_ids:
                    # [FIX] 跳过当前节点自己
                    # 这样当前节点的 inputs (如 grid_size, font_size) 变化不会改变 content_hash
                    if ignore_id is not None and str(nid) == str(ignore_id):
                        continue

                    node = prompt[nid]
                    class_type = node.get("class_type", "")
                    inputs = node.get("inputs", {})
                    
                    # 将 inputs 序列化为字符串。
                    try:
                        inputs_str = json.dumps(inputs, sort_keys=True, default=str)
                    except:
                        inputs_str = str(inputs)
                        
                    content_str += f"|{nid}:{class_type}:{inputs_str}"
            except Exception as e:
                print(f"[XY_Image] Hash calculation warning: {e}")
                content_str += str(prompt)
                    
        return hashlib.md5(content_str.encode('utf-8')).hexdigest()

    @classmethod
    def IS_CHANGED(cls, 间隔大小, 字体大小, 展示面板, x_attr, y_attr, images=None, **kwargs):
        """
        告诉 ComfyUI 何时需要执行节点。
        只要 样式(grid/font/layout) 或 内容(content_hash) 变了，都返回新值以触发执行。
        """
        prompt = kwargs.get("prompt", None)
        my_unique_id = kwargs.get("my_unique_id", None)
        
        # 计算哈希时排除自己 (my_unique_id)
        content_hash = cls._calculate_content_hash(x_attr, y_attr, prompt, my_unique_id)
        
        # 返回组合哈希：内容 + 样式 + 布局
        return f"{content_hash}_{间隔大小}_{字体大小}_{展示面板}"

    def _parse_axis_config(self, text: str) -> Dict[str, Any]:
        """解析格式: <axis_id:label> [node:input]='value'"""
        if not text or not isinstance(text, str):
            return None
        
        result = {}
        try:
            text_clean = text.strip()
            parts = text_clean.split('<')
            current_axis_id = None
            
            for part in parts:
                part = part.strip()
                if not part: continue
                
                if '>' in part:
                    header, body = part.split('>', 1)
                    if ':' in header:
                        axis_id, label = header.split(':', 1)
                        axis_id = axis_id.strip()
                        label = label.strip()
                        current_axis_id = axis_id
                        result[axis_id] = {"label": label}
                        self._parse_params(body, result, axis_id)
                else:
                    if current_axis_id:
                        self._parse_params(part, result, current_axis_id)
            return result if result else None
        except Exception as e:
            print(f"[XY_Image] Parse error: {e}")
            traceback.print_exc()
            return None

    def _parse_params(self, text, result, axis_id):
        pattern = r'\[(\d+):([^\]]+)\]\s*=\s*["\']?([^\r\n"\']*)["\'"]?'
        matches = re.findall(pattern, text)
        for node_id, input_name, value in matches:
            node_id = node_id.strip()
            input_name = input_name.strip()
            value = value.strip()
            if node_id not in result[axis_id]:
                result[axis_id][node_id] = {}
            result[axis_id][node_id][input_name] = value

    def _create_grid_image(self, tensor_list, x_labels, y_labels, grid_spacing=10, font_size=32, layout="XY Plot"):
        """拼接网格图，支持多种布局模式"""
        if not tensor_list: return None
        
        pil_images = []
        for t in tensor_list:
            if t.ndim == 4: t = t.squeeze(0)
            np_img = (t.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pil_images.append(Image.fromarray(np_img))
        
        if not pil_images: return None
        w, h = pil_images[0].size
        cols = len(x_labels)
        rows = len(y_labels)
        
        # 字体处理
        font = None
        try:
            font_paths = [
                "simhei.ttf", "simsun.ttc", "Microsoft YaHei.ttf", "msyh.ttf",
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/STHeiti Light.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "arial.ttf", "Arial.ttf"
            ]
            for path in font_paths:
                try:
                    font = ImageFont.truetype(path, font_size)
                    break
                except: continue
            if font is None: font = ImageFont.load_default()
        except: font = ImageFont.load_default()

        # 公共参数
        outer_margin = 20
        
        # 根据布局计算尺寸
        if layout == "XY Plot":
            # 传统表格模式
            header_h = int(font_size * 2) + 10
            max_y_len = max([len(str(l)) for l in y_labels]) if y_labels else 0
            sidebar_w = int(max_y_len * font_size * 1.0) + 40
            if sidebar_w < 80: sidebar_w = 80
            
            grid_w = sidebar_w + cols * w + (cols - 1) * grid_spacing + outer_margin * 2
            grid_h = header_h + rows * h + (rows - 1) * grid_spacing + outer_margin * 2
            
        else:
            # X:Y 或 Y:X 模式 (标签在每张图下方)
            label_height = int(font_size * 1.5) + 10 # 文字区域高度
            
            grid_w = cols * w + (cols - 1) * grid_spacing + outer_margin * 2
            grid_h = rows * (h + label_height) + (rows - 1) * grid_spacing + outer_margin * 2

        grid_img = Image.new('RGB', (grid_w, grid_h), color='white')
        draw = ImageDraw.Draw(grid_img)

        def draw_centered_text(draw, text, cx, cy, font, fill="black"):
            try:
                left, top, right, bottom = draw.textbbox((0, 0), str(text), font=font)
                text_w = right - left
                text_h = bottom - top
            except:
                text_w, text_h = draw.textsize(str(text), font=font)
            draw.text((cx - text_w / 2, cy - text_h / 2), str(text), fill=fill, font=font)

        if layout == "XY Plot":
            # 1. 绘制 X 轴标签
            for i, label in enumerate(x_labels):
                img_x_start = sidebar_w + outer_margin + i * (w + grid_spacing)
                cx = img_x_start + w // 2
                cy = outer_margin + header_h // 2
                draw_centered_text(draw, label, cx, cy, font)
            
            # 2. 绘制 Y 轴标签
            for j, label in enumerate(y_labels):
                img_y_start = header_h + outer_margin + j * (h + grid_spacing)
                cx = (sidebar_w + outer_margin) // 2
                cy = img_y_start + h // 2
                draw_centered_text(draw, label, cx, cy, font)

            # 3. 粘贴图片
            for idx, img in enumerate(pil_images):
                row = idx // cols
                col = idx % cols
                if row >= rows: break
                
                x = sidebar_w + outer_margin + col * (w + grid_spacing)
                y = header_h + outer_margin + row * (h + grid_spacing)
                
                if img.size != (w, h): img = img.resize((w, h))
                grid_img.paste(img, (x, y))
        else:
            # 列表模式 (X:Y / Y:X)
            label_height = int(font_size * 1.5) + 10
            
            for idx, img in enumerate(pil_images):
                row = idx // cols
                col = idx % cols
                if row >= rows: break
                
                # 计算位置
                x = outer_margin + col * (w + grid_spacing)
                y = outer_margin + row * (h + label_height + grid_spacing)
                
                if img.size != (w, h): img = img.resize((w, h))
                grid_img.paste(img, (x, y))
                
                # 绘制下方标签
                x_str = str(x_labels[col]) if col < len(x_labels) else ""
                y_str = str(y_labels[row]) if row < len(y_labels) else ""
                
                if layout == "X:Y":
                    label_text = f"{x_str} : {y_str}"
                else:
                    label_text = f"{y_str} : {x_str}"
                
                cx = x + w // 2
                cy = y + h + label_height // 2 # 位于图片下方的文字区域中心
                draw_centered_text(draw, label_text, cx, cy, font)

        np_grid = np.array(grid_img).astype(np.float32) / 255.0
        torch_grid = torch.from_numpy(np_grid).unsqueeze(0)
        return torch_grid

    # --- 辅助工具：类型转换 ---
    def _convert_type(self, value, type_name):
        try:
            if type_name == 'INT': return int(float(str(value)))
            elif type_name == 'FLOAT': return float(value)
            elif type_name in ['BOOL', 'BOOLEAN']:
                if isinstance(value, str): return value.lower() in ('true', '1', 'yes', 'on')
                return bool(value)
            return value
        except: return value

    # --- 辅助工具：应用修改 ---
    def _apply_modifications(self, p, modifications, node_class_mappings):
        if not modifications: return p
        
        for node_id_str, inputs_dict in modifications.items():
            if node_id_str == 'label': continue
            
            target_key = None
            if node_id_str in p: target_key = node_id_str
            else:
                for k in p.keys():
                    if str(k) == str(node_id_str):
                        target_key = k
                        break
            
            if not target_key: continue
            
            try:
                node = p[target_key]
                class_type = node.get("class_type", "")
                node_inputs = node.get("inputs", {})
                
                class_def = node_class_mappings.get(class_type)
                input_types = class_def.INPUT_TYPES() if class_def else {}
                
                for input_name, value in inputs_dict.items():
                    if input_name == 'label': continue
                    
                    type_name = None
                    if input_types:
                        for cat in ['required', 'optional']:
                            if input_name in input_types.get(cat, {}):
                                val_def = input_types[cat][input_name]
                                if isinstance(val_def, tuple) and len(val_def) > 0:
                                    type_name = val_def[0]
                                    break
                    
                    if type_name:
                        node_inputs[input_name] = self._convert_type(value, type_name)
                    else:
                        node_inputs[input_name] = value
                        
            except Exception as e:
                print(f"[XY_Image] Apply error: {e}")
        return p

    # --- 执行节点逻辑 ---
    def _execute_node_logic(self, node_id, p_copy, iter_outputs, node_class_mappings):
        node_id = str(node_id)
        if node_id in iter_outputs: return True
        if node_id not in p_copy: return False
        
        node_data = p_copy[node_id]
        class_type = node_data.get("class_type")
        
        # 依赖递归
        inputs = node_data.get("inputs", {})
        for v in inputs.values():
            if isinstance(v, list) and len(v) > 0:
                if not self._execute_node_logic(str(v[0]), p_copy, iter_outputs, node_class_mappings): 
                    return False
        
        # 准备输入
        func_inputs = {}
        node_cls = node_class_mappings.get(class_type)
        if not node_cls: return False
        
        obj = node_cls()
        
        for k, v in inputs.items():
            if isinstance(v, list) and len(v) >= 2:
                from_id = str(v[0])
                from_idx = int(v[1])
                
                if from_id in iter_outputs:
                    outputs_tuple = iter_outputs[from_id]
                    
                    is_container = isinstance(outputs_tuple, tuple)
                    if not is_container:
                        if hasattr(outputs_tuple, 'args') and hasattr(outputs_tuple, '__getitem__') and not isinstance(outputs_tuple, (dict, torch.Tensor)):
                            is_container = True
                    
                    val = None
                    if is_container:
                        try:
                            length = len(outputs_tuple) if isinstance(outputs_tuple, tuple) else len(outputs_tuple.args)
                            val = outputs_tuple[from_idx] if from_idx < length else outputs_tuple[0]
                        except:
                            try: val = outputs_tuple[from_idx]
                            except: val = outputs_tuple 
                    else:
                        val = outputs_tuple
                    
                    func_inputs[k] = val
                else: return False
            else:
                func_inputs[k] = v
        
        # 执行
        try:
            func = getattr(obj, obj.FUNCTION)
            res = func(**func_inputs)
            
            is_container = isinstance(res, tuple)
            if not is_container:
                if hasattr(res, 'args') and hasattr(res, '__getitem__') and not isinstance(res, (dict, torch.Tensor)):
                    is_container = True

            if not is_container: res = (res,)
            iter_outputs[node_id] = res
            return True
        except Exception as e:
            print(f"[XY_Image] Node {node_id} ({class_type}) failed: {e}")
            traceback.print_exc()
            return False

    def build_grid(self, images, 间隔大小, 字体大小, 展示面板, x_attr, y_attr, prompt=None, extra_pnginfo=None, my_unique_id=None):
        global _XY_GLOBAL_CACHE
        print(f"\n[XY_Image] ========== START ==========")
        
        # 变量重命名，保持内部逻辑一致
        grid_size = 间隔大小
        font_size = 字体大小
        display_layout = 展示面板

        if comfy_nodes is None:
            return images, images

        if not isinstance(prompt, dict):
            return images, images
        
        # 1. 计算 Content Hash (不包含 grid_size/font_size/layout)
        # [FIX] 传入 my_unique_id 以便排除自身参数对哈希的影响
        current_content_hash = self._calculate_content_hash(x_attr, y_attr, prompt, my_unique_id)
        
        # 解析参数（无论是否缓存都需要，用于labels）
        x_points = self._parse_axis_config(x_attr) if x_attr else None
        y_points = self._parse_axis_config(y_attr) if y_attr else None
        x_keys = list(x_points.keys()) if x_points else [None]
        y_keys = list(y_points.keys()) if y_points else [None]
        x_labels = [str(x_points[k].get("label", k)) if x_points and k in x_points else "" for k in x_keys]
        y_labels = [str(y_points[k].get("label", k)) if y_points and k in y_points else "" for k in y_keys]

        result_tensors = None

        # 2. 检查缓存
        if current_content_hash in _XY_GLOBAL_CACHE:
            print(f"[XY_Image] Cache HIT! Content unchanged, reusing images. Only updating grid style.")
            result_tensors = _XY_GLOBAL_CACHE[current_content_hash]
        else:
            print(f"[XY_Image] Cache MISS or Content Changed. Executing workflow...")
            result_tensors = []
            total = len(x_keys) * len(y_keys)
            iteration = 0
            
            NODE_CLASS_MAPPINGS = comfy_nodes.NODE_CLASS_MAPPINGS

            # 循环生成
            for yi, y_key in enumerate(y_keys):
                y_mod = y_points.get(y_key, {}) if y_points else {}
                for xi, x_key in enumerate(x_keys):
                    iteration += 1
                    x_mod = x_points.get(x_key, {}) if x_points else {}
                    
                    print(f"[XY_Image] Iteration {iteration}/{total} (Y={yi+1}, X={xi+1})")
                    
                    try:
                        p_copy = copy.deepcopy(prompt)
                        if x_mod: p_copy = self._apply_modifications(p_copy, x_mod, NODE_CLASS_MAPPINGS)
                        if y_mod: p_copy = self._apply_modifications(p_copy, y_mod, NODE_CLASS_MAPPINGS)
                        
                        # 查找输出节点
                        all_ids = set(p_copy.keys())
                        used_ids = set()
                        for n in p_copy.values():
                            for inp in n.get("inputs", {}).values():
                                if isinstance(inp, list) and len(inp) > 0: used_ids.add(str(inp[0]))
                        target_nodes = list(all_ids - used_ids)
                        
                        # 执行
                        iter_outputs = {}
                        for t in target_nodes:
                            self._execute_node_logic(t, p_copy, iter_outputs, NODE_CLASS_MAPPINGS)
                        
                        # 搜刮图片
                        found_img = None
                        sorted_ids = sorted(list(iter_outputs.keys()), key=lambda x: int(x) if x.isdigit() else 0, reverse=True)
                        
                        def get_all_items(container):
                            if isinstance(container, tuple): return list(container)
                            if hasattr(container, 'args'): return list(container.args) if hasattr(container.args, '__iter__') else [container.args]
                            return [container]

                        for nid in sorted_ids:
                            outs = iter_outputs[nid]
                            try:
                                candidates = get_all_items(outs)
                                for item in candidates:
                                    if isinstance(item, torch.Tensor):
                                        if item.ndim == 4 and item.shape[-1] in [3, 4]:
                                            found_img = item; break
                                        if item.ndim == 3 and item.shape[-1] in [3, 4]:
                                            found_img = item.unsqueeze(0); break
                                if found_img is not None: break
                            except: pass
                        
                        if found_img is not None:
                            result_tensors.append(found_img.cpu())
                        else:
                            print(f"[XY_Image] Warning: No image found in iteration {iteration}")
                            if isinstance(images, torch.Tensor):
                                 result_tensors.append(images[0].unsqueeze(0).cpu())
                                 
                    except Exception as e:
                        print(f"[XY_Image] Iteration failed: {e}")
                        traceback.print_exc()
            
            # 更新缓存 (清空旧的，只留最新的)
            if result_tensors:
                _XY_GLOBAL_CACHE.clear()
                _XY_GLOBAL_CACHE[current_content_hash] = result_tensors

        # 3. 无论是否命中缓存，都要根据当前的 grid_size/font_size/layout 重新生成网格图
        grid_tensor = None
        try:
            if result_tensors:
                print(f"[XY_Image] Generating grid image (Gap={grid_size}, Font={font_size}, Layout={display_layout})...")
                # 传入 layout 参数
                grid_tensor = self._create_grid_image(result_tensors, x_labels, y_labels, grid_size, font_size, display_layout)
        except Exception as e:
            print(f"[XY_Image] Grid generation failed: {e}")
            traceback.print_exc()

        if grid_tensor is None: grid_tensor = images

        # 拼接 Images
        try:
            final_batch = torch.cat(result_tensors, dim=0) if result_tensors else images
        except:
            final_batch = result_tensors[0] if result_tensors else images
            
        print(f"[XY_Image] Done.")
        return final_batch, grid_tensor

NODE_CLASS_MAPPINGS = {
    "XY_Image": XYImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XY_Image": "XY测试图☀"
}