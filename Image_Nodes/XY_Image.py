import json
import copy
import traceback
import re
from typing import Any, Dict, List
import sys
import os
import uuid
import hashlib
import gc
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import nodes as comfy_nodes
import execution
import comfy.model_management

# --- 全局缓存 ---
# 结构: { "content_hash_string": [tensor1, tensor2, ...] }
_XY_GLOBAL_CACHE = {}

class XYImage:
    """
    独立实现的 XY Plot 节点。
    支持：
    1. 智能缓存：只重跑参数变化的节点（极大提升速度）。
    2. 样式缓存：只改网格/字体时不重跑。
    3. 多种布局：XY Plot, X:Y, Y:X。
    4. 显存优化：主动释放显存，防止 LoRA 切换时卡顿。
    5. [修复] 能够从 SaveImage/PreviewImage 等非 Tensor 输出节点的上游抓取图片。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "gap_size": ("INT", {"default": 10, "min": 5, "max": 1000, "step": 5}),
                "font_size": ("INT", {"default": 50, "min": 50, "max": 250, "step": 1}),
                "panel_mode": (["XY Plot", "X:Y", "Y:X"],),
                "x_attr": ("STRING", {"default": "", "multiline": True, "placeholder": "右键唤起参数浮窗\nX轴配置\nRight-click to open the parameter pop-up window\nX-axis configuration: <x:Label>[node_id:input]='value'"}),
                "y_attr": ("STRING", {"default": "", "multiline": True, "placeholder": "右键唤起参数浮窗\nY轴配置\nRight-click to open the parameter pop-up window\nY-axis configuration: <y:Label>[node_id:input]='value'"}),
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

    # --- 智能检测：是否需要响应 ComfyUI 的执行请求 ---
    @classmethod
    def IS_CHANGED(cls, gap_size, font_size, panel_mode, x_attr, y_attr, images=None, **kwargs):
        prompt = kwargs.get("prompt", None)
        my_unique_id = kwargs.get("my_unique_id", None)
        content_hash = cls._calculate_workflow_fingerprint(x_attr, y_attr, prompt, my_unique_id)
        return f"{content_hash}_{gap_size}_{font_size}_{panel_mode}"

    @classmethod
    def _calculate_workflow_fingerprint(cls, x_attr, y_attr, prompt, ignore_id=None):
        """计算整个工作流的指纹，用于判断是否需要重跑"""
        content_str = f"{x_attr}_{y_attr}"
        if prompt:
            try:
                node_ids = sorted(list(prompt.keys()))
                for nid in node_ids:
                    if ignore_id is not None and str(nid) == str(ignore_id): continue
                    node = prompt[nid]
                    inputs = node.get("inputs", {})
                    # [Opt] 使用 json.dumps 保证字典顺序一致，避免哈希抖动
                    try: inputs_str = json.dumps(inputs, sort_keys=True, default=str)
                    except: inputs_str = str(inputs)
                    content_str += f"|{nid}:{node.get('class_type','')}:{inputs_str}"
            except: pass
        return hashlib.md5(content_str.encode('utf-8')).hexdigest()

    # --- 执行引擎核心方法 ---

    def _extract_value(self, outputs_tuple, slot_index):
        """从节点输出元组中提取特定插槽的值"""
        if outputs_tuple is None: return None
        
        # 智能识别 WrappedTuple (某些 ComfyUI 环境)
        is_container = isinstance(outputs_tuple, tuple)
        if not is_container:
            if hasattr(outputs_tuple, 'args') and hasattr(outputs_tuple, '__getitem__') and not isinstance(outputs_tuple, (dict, torch.Tensor)):
                is_container = True
        
        if is_container:
            try:
                # 获取长度
                length = len(outputs_tuple) if isinstance(outputs_tuple, tuple) else len(outputs_tuple.args)
                if slot_index < length:
                    return outputs_tuple[slot_index]
                return outputs_tuple[0] # Fallback
            except:
                return outputs_tuple
        return outputs_tuple

    def _safe_to_cpu(self, obj):
        """[Opt] 递归将对象中的 Tensor 移动到 CPU 以节省显存"""
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        elif isinstance(obj, dict):
            # 处理 Latent 字典
            return {k: self._safe_to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._safe_to_cpu(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._safe_to_cpu(v) for v in obj)
        return obj

    def _execute_single_node(self, node_class_mappings, class_type, func_inputs):
        """实例化并执行单个节点"""
        node_cls = node_class_mappings.get(class_type)
        if not node_cls: return None
        
        try:
            obj = node_cls()
            func = getattr(obj, obj.FUNCTION)
            res = func(**func_inputs)
            
            # 统一包装为 tuple 以便后续处理
            is_container = isinstance(res, tuple)
            if not is_container:
                if hasattr(res, 'args') and hasattr(res, '__getitem__') and not isinstance(res, (dict, torch.Tensor)):
                    is_container = True
            
            if not is_container: res = (res,)
            return res
        except Exception as e:
            # [Opt] 捕获错误但不崩溃，允许部分失败 (如 ShowText 节点报错)
            print(f"[XY_Image] Warning: Node {class_type} execution failed: {e}")
            return None

    def _get_node_hash_and_output(self, node_id, prompt, batch_cache, current_step_cache, node_class_mappings):
        """
        递归获取节点的 Hash 和 Output。
        如果 Hash 在 batch_cache 中存在，则直接返回缓存结果（跳过执行）。
        """
        node_id = str(node_id)
        
        # 1. 检查本次迭代的递归缓存 (避免环路和重复计算)
        if node_id in current_step_cache:
            return current_step_cache[node_id]

        if node_id not in prompt:
            return None, None

        node_data = prompt[node_id]
        class_type = node_data.get("class_type")
        inputs = node_data.get("inputs", {})

        # 2. 递归处理依赖，并计算当前节点的输入指纹
        input_hashes = []
        func_inputs = {}
        
        # 排序以保证 Hash 确定性
        sorted_keys = sorted(inputs.keys())
        
        for k in sorted_keys:
            v = inputs[k]
            # 检查是否为链接依赖: ["node_id", slot_index]
            if isinstance(v, list) and len(v) == 2 and (isinstance(v[0], str) or isinstance(v[0], int)):
                dep_id = str(v[0])
                dep_slot = int(v[1])
                
                # 递归调用
                dep_hash, dep_outputs = self._get_node_hash_and_output(dep_id, prompt, batch_cache, current_step_cache, node_class_mappings)
                
                if dep_hash is None: 
                    # 上游失败
                    return None, None
                
                # 将上游的 Hash 加入输入指纹 (而不是上游的值，值可能很大)
                input_hashes.append(f"{k}:{dep_hash}:{dep_slot}")
                
                # 提取实际值传给函数
                val = self._extract_value(dep_outputs, dep_slot)
                func_inputs[k] = val
            else:
                # 字面量输入
                # [Opt] 使用 json.dumps 确保复杂结构(如List/Dict)的Hash稳定性
                try: val_str = json.dumps(v, sort_keys=True, default=str)
                except: val_str = str(v)
                
                # 简易压缩长字符串
                if len(val_str) > 100: val_str = hashlib.md5(val_str.encode()).hexdigest()
                input_hashes.append(f"{k}:{val_str}")
                func_inputs[k] = v

        # 3. 计算当前节点的唯一指纹 (Class + Inputs Hash)
        # 只要上游任何节点的参数变了，dep_hash 就会变，进而导致当前 node_hash 变，触发重跑
        hash_src = f"{class_type}|{','.join(input_hashes)}"
        node_hash = hashlib.md5(hash_src.encode('utf-8')).hexdigest()

        # 4. 检查全局 Batch 缓存
        if node_hash in batch_cache:
            # 【缓存命中】直接使用之前算好的结果！
            output = batch_cache[node_hash]
        else:
            # 【缓存未命中】必须执行
            output = self._execute_single_node(node_class_mappings, class_type, func_inputs)
            if output is not None:
                # [Opt] 关键优化：存入缓存前将 Tensor 移至 CPU，防止 VRAM 挤兑
                # 注意：ModelPatcher 等对象不受此影响，因为它们不是 Tensor
                cached_output = self._safe_to_cpu(output)
                batch_cache[node_hash] = cached_output
                output = cached_output

        # 5. 存入步骤缓存
        current_step_cache[node_id] = (node_hash, output)
        return node_hash, output

    # --- 辅助方法 ---

    def _parse_axis_config(self, text: str) -> Dict[str, Any]:
        if not text or not isinstance(text, str): return None
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
                        axis_id = axis_id.strip(); label = label.strip()
                        current_axis_id = axis_id
                        result[axis_id] = {"label": label}
                        self._parse_params(body, result, axis_id)
                else:
                    if current_axis_id: self._parse_params(part, result, current_axis_id)
            return result if result else None
        except: return None

    def _parse_params(self, text, result, axis_id):
        pattern = r'\[(\d+):([^\]]+)\]\s*=\s*["\']?([^\r\n"\']*)["\'"]?'
        matches = re.findall(pattern, text)
        for node_id, input_name, value in matches:
            node_id = node_id.strip(); input_name = input_name.strip(); value = value.strip()
            if node_id not in result[axis_id]: result[axis_id][node_id] = {}
            result[axis_id][node_id][input_name] = value

    def _convert_type(self, value, type_name):
        try:
            if type_name == 'INT': return int(float(str(value)))
            elif type_name == 'FLOAT': return float(value)
            elif type_name in ['BOOL', 'BOOLEAN']:
                if isinstance(value, str): return value.lower() in ('true', '1', 'yes', 'on')
                return bool(value)
            return value
        except: return value

    def _apply_modifications(self, p, modifications, node_class_mappings):
        if not modifications: return p
        for node_id_str, inputs_dict in modifications.items():
            if node_id_str == 'label': continue
            target_key = None
            if node_id_str in p: target_key = node_id_str
            else:
                for k in p.keys():
                    if str(k) == str(node_id_str): target_key = k; break
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
                                if isinstance(val_def, tuple) and len(val_def) > 0: type_name = val_def[0]; break
                    if type_name: node_inputs[input_name] = self._convert_type(value, type_name)
                    else: node_inputs[input_name] = value
            except: pass
        return p

    def _create_grid_image(self, tensor_list, x_labels, y_labels, grid_spacing=10, font_size=32, layout="XY Plot"):
        if not tensor_list: return None
        pil_images = []
        for t in tensor_list:
            if t.ndim == 4: t = t.squeeze(0)
            np_img = (t.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pil_images.append(Image.fromarray(np_img))
        if not pil_images: return None
        w, h = pil_images[0].size
        cols = len(x_labels); rows = len(y_labels)
        
        font = None
        try:
            font_paths = ["simhei.ttf", "Microsoft YaHei.ttf", "arial.ttf", "/System/Library/Fonts/PingFang.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]
            for path in font_paths:
                try: font = ImageFont.truetype(path, font_size); break
                except: continue
            if font is None: font = ImageFont.load_default()
        except: font = ImageFont.load_default()

        outer_margin = 20
        if layout == "XY Plot":
            header_h = int(font_size * 2) + 10
            max_y_len = max([len(str(l)) for l in y_labels]) if y_labels else 0
            sidebar_w = int(max_y_len * font_size * 1.0) + 40
            if sidebar_w < 80: sidebar_w = 80
            grid_w = sidebar_w + cols * w + (cols - 1) * grid_spacing + outer_margin * 2
            grid_h = header_h + rows * h + (rows - 1) * grid_spacing + outer_margin * 2
        else:
            label_height = int(font_size * 1.5) + 10
            grid_w = cols * w + (cols - 1) * grid_spacing + outer_margin * 2
            grid_h = rows * (h + label_height) + (rows - 1) * grid_spacing + outer_margin * 2

        grid_img = Image.new('RGB', (grid_w, grid_h), color='white')
        draw = ImageDraw.Draw(grid_img)

        def draw_text(draw, text, cx, cy, font):
            try:
                left, top, right, bottom = draw.textbbox((0, 0), str(text), font=font)
                w, h = right - left, bottom - top
            except: w, h = draw.textsize(str(text), font=font)
            draw.text((cx - w / 2, cy - h / 2), str(text), fill="black", font=font)

        if layout == "XY Plot":
            for i, l in enumerate(x_labels):
                cx = sidebar_w + outer_margin + i * (w + grid_spacing) + w // 2
                draw_text(draw, l, cx, outer_margin + header_h // 2, font)
            for j, l in enumerate(y_labels):
                cy = header_h + outer_margin + j * (h + grid_spacing) + h // 2
                draw_text(draw, l, (sidebar_w + outer_margin) // 2, cy, font)
            for idx, img in enumerate(pil_images):
                r = idx // cols; c = idx % cols
                if r >= rows: break
                if img.size != (w, h): img = img.resize((w, h))
                grid_img.paste(img, (sidebar_w + outer_margin + c * (w + grid_spacing), header_h + outer_margin + r * (h + grid_spacing)))
        else:
            for idx, img in enumerate(pil_images):
                r = idx // cols; c = idx % cols
                if r >= rows: break
                x = outer_margin + c * (w + grid_spacing)
                y = outer_margin + r * (h + label_height + grid_spacing)
                if img.size != (w, h): img = img.resize((w, h))
                grid_img.paste(img, (x, y))
                lbl = f"{x_labels[c]} : {y_labels[r]}" if layout == "X:Y" else f"{y_labels[r]} : {x_labels[c]}"
                draw_text(draw, lbl, x + w // 2, y + h + label_height // 2, font)

        np_grid = np.array(grid_img).astype(np.float32) / 255.0
        return torch.from_numpy(np_grid).unsqueeze(0)

    # --- 主入口 ---
    def build_grid(self, images, gap_size, font_size, panel_mode, x_attr, y_attr, prompt=None, extra_pnginfo=None, my_unique_id=None):
        global _XY_GLOBAL_CACHE
        print(f"\n[XY_Image] ========== START ==========")
        
        if comfy_nodes is None or not isinstance(prompt, dict):
            return images, images
        
        # 1. 计算 Content Hash
        current_content_hash = self._calculate_workflow_fingerprint(x_attr, y_attr, prompt, my_unique_id)
        
        x_points = self._parse_axis_config(x_attr) if x_attr else None
        y_points = self._parse_axis_config(y_attr) if y_attr else None
        x_keys = list(x_points.keys()) if x_points else [None]
        y_keys = list(y_points.keys()) if y_points else [None]
        x_labels = [str(x_points[k].get("label", k)) if x_points and k in x_points else "" for k in x_keys]
        y_labels = [str(y_points[k].get("label", k)) if y_points and k in y_points else "" for k in y_keys]

        result_tensors = None

        # 2. 检查缓存是否命中
        if current_content_hash in _XY_GLOBAL_CACHE:
            print(f"[XY_Image] Content Cache HIT! Skipping workflow execution.")
            result_tensors = _XY_GLOBAL_CACHE[current_content_hash]
        else:
            print(f"[XY_Image] Content Cache MISS. Starting Smart Execution...")
            result_tensors = []
            
            # 这是一个仅存在于本次 build_grid 执行周期的缓存
            # 用于在 X/Y 循环之间共享节点结果 (如模型加载)
            # Key: Node_Hash, Value: Output
            batch_execution_cache = {} 
            
            NODE_CLASS_MAPPINGS = comfy_nodes.NODE_CLASS_MAPPINGS
            total = len(x_keys) * len(y_keys)
            iteration = 0
            
            # 初始化历史记录，用于过滤重复日志
            last_mods_log = {'x': None, 'y': None}

            for yi, y_key in enumerate(y_keys):
                y_mod = y_points.get(y_key, {}) if y_points else {}
                for xi, x_key in enumerate(x_keys):
                    iteration += 1
                    x_mod = x_points.get(x_key, {}) if x_points else {}
                    
                    print(f"[XY_Image] Iteration {iteration}/{total} (Y={yi+1}, X={xi+1})")
                    
                    # --- 新增：打印本次运行变动的参数 (优化格式 & 去重) ---
                    current_log_parts = []
                    
                    def get_clean_mod_str(mod_dict):
                        items = []
                        if not mod_dict: return ""
                        # 排序保证一致性
                        for nid in sorted(mod_dict.keys()):
                            if nid == 'label': continue
                            params = mod_dict[nid]
                            if isinstance(params, dict):
                                for k, v in params.items():
                                    items.append(f"[{nid}].{k}={v}")
                        return ", ".join(items)

                    x_str = get_clean_mod_str(x_mod)
                    y_str = get_clean_mod_str(y_mod)
                    
                    # 对比上次打印的内容，只有变动了才打印
                    # 分别跟踪 X 和 Y，避免同一行内 Y 参数重复刷屏
                    if x_str and x_str != last_mods_log['x']:
                        current_log_parts.append(f"X: {x_str}")
                        last_mods_log['x'] = x_str
                    
                    if y_str and y_str != last_mods_log['y']:
                        current_log_parts.append(f"Y: {y_str}")
                        last_mods_log['y'] = y_str
                        
                    if current_log_parts:
                        print(f"[XY_Image] Params: {' | '.join(current_log_parts)}")
                    # ------------------------------------
                    
                    try:
                        p_copy = copy.deepcopy(prompt)
                        if x_mod: p_copy = self._apply_modifications(p_copy, x_mod, NODE_CLASS_MAPPINGS)
                        if y_mod: p_copy = self._apply_modifications(p_copy, y_mod, NODE_CLASS_MAPPINGS)
                        
                        # 查找输出节点 (Leaves)
                        all_ids = set(p_copy.keys())
                        used_ids = set()
                        for n in p_copy.values():
                            for inp in n.get("inputs", {}).values():
                                if isinstance(inp, list) and len(inp) > 0: used_ids.add(str(inp[0]))
                        # 排除自己(my_unique_id)以免死循环
                        if my_unique_id: used_ids.add(str(my_unique_id))
                        
                        target_nodes = list(all_ids - used_ids)
                        
                        # 递归执行，带有 batch_execution_cache
                        current_step_cache = {} # 防止单次图中的环路
                        
                        # 执行所有叶子节点（这将触发整图执行）
                        for t in target_nodes:
                            self._get_node_hash_and_output(t, p_copy, batch_execution_cache, current_step_cache, NODE_CLASS_MAPPINGS)
                        
                        # --- 搜刮图片 (FIXED LOGIC) ---
                        # 原版代码只在 iter_outputs (叶子节点输出) 中找图片，导致 SaveImage (返回 Dict) 无法提供图片。
                        # 现在我们在 current_step_cache (所有已执行节点) 中找图片。
                        
                        found_img = None
                        
                        # 按 Node ID 倒序搜索 (通常后面的 ID 是生成结果)
                        sorted_cache_ids = sorted(list(current_step_cache.keys()), key=lambda x: int(x) if str(x).isdigit() else 0, reverse=True)
                        
                        def get_items(c):
                            if isinstance(c, tuple): return list(c)
                            if hasattr(c, 'args'): return list(c.args) if hasattr(c.args, '__iter__') else [c.args]
                            return [c]

                        for nid in sorted_cache_ids:
                            # cache 结构: { node_id: (hash, output_tuple) }
                            _, outs = current_step_cache[nid]
                            try:
                                candidates = get_items(outs)
                                for item in candidates:
                                    if isinstance(item, torch.Tensor):
                                        # 检查是否为图片张量 [B, H, W, C] 或 [B, C, H, W]
                                        # ComfyUI 标准图片是 [B, H, W, 3]
                                        if item.ndim == 4 and item.shape[-1] in [3, 4]:
                                            found_img = item; break
                                        # 某些特殊节点可能是 [C, H, W] ? 兼容性处理
                                        if item.ndim == 3 and item.shape[-1] in [3, 4]:
                                            found_img = item.unsqueeze(0); break
                                if found_img is not None: break
                            except: pass
                        
                        if found_img is not None:
                            # [Opt] 将搜刮到的最终图片也放到 CPU
                            result_tensors.append(found_img.cpu())
                        else:
                            print(f"[XY_Image] Warning: No image found in iteration {iteration}")
                            # Fallback: 使用输入占位图
                            if isinstance(images, torch.Tensor):
                                 result_tensors.append(images[0].unsqueeze(0).cpu())
                    
                    except Exception as e:
                        print(f"[XY_Image] Iteration failed: {e}")
                        traceback.print_exc()
            
                # [Crucial Optimization] 每次循环结束后，主动清理显存
                # 这会移除当前循环中加载的临时模型（如被替换掉的 LoRA），防止显存挤兑
                try:
                    gc.collect()
                    comfy.model_management.soft_empty_cache()
                except:
                    pass

            if result_tensors:
                _XY_GLOBAL_CACHE.clear()
                _XY_GLOBAL_CACHE[current_content_hash] = result_tensors

        # 3. 拼图
        grid_tensor = None
        try:
            if result_tensors:
                print(f"[XY_Image] Generating grid: {panel_mode}...")
                grid_tensor = self._create_grid_image(result_tensors, x_labels, y_labels, gap_size, font_size, panel_mode)
        except Exception as e:
            print(f"[XY_Image] Grid generation failed: {e}")
            traceback.print_exc()

        if grid_tensor is None: grid_tensor = images

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
    "XY_Image": "XYImage☀"
}
