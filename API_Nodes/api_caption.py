# ComfyUI 节点：api_caption（精简版）
import base64
import io
import json
from typing import Tuple

import numpy as np
import requests
from PIL import Image
import torch


def tensor2pil(t_image: torch.Tensor) -> Image.Image:
    if not isinstance(t_image, torch.Tensor):
        raise TypeError("输入必须为 torch.Tensor")
    arr = t_image.cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.clip(255.0 * arr.squeeze(), 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    return Image.fromarray(arr)


class api_caption:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "API类型": (["硅基流动", "贞贞工坊", "OpenRouter", "其他"], {"default": "硅基流动"}),
                "请求地址": ("STRING", {"default": "<url>"}),
                "API_Key": ("STRING", {"default": "<your_key>"}),
                "模型名称": ("STRING", {"default": "Qwen/Qwen3-VL-32B-Instruct"}),
                "image": ("IMAGE",),
                "提示词": ("STRING", {"default": "你是一位专业的AI图像生成提示词工程师。请详细描述这张图像的主体、前景、中景、背景、构图、视觉引导、色调、光影氛围等细节并创作出具有深度、氛围和艺术感的图像提示词。要求：中文提示词，不要出现对图像水印的描述，不要出现无关的文字和符号，不需要总结，限制在800字以内", "multiline": True, "rows": 4}),
                "输出语言": (["中文", "英文"], {"default": "中文"}),
                "温度": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "DaNodes/API"

    def generate(self, API类型: str, API_Key: str, 模型名称: str, image: torch.Tensor, 提示词: str,
                 输出语言: str, 温度: float = 0.5, 请求地址: str = "") -> Tuple[str]:
        try:
            # 根据 provider 选择实际的 URL
            provider_map = {
                "OpenRouter": "https://openrouter.ai/api/v1/chat/completions",
                "硅基流动": "https://api.siliconflow.cn/v1/chat/completions",
                "贞贞工坊": "https://api.bltcy.ai/v1/chat/completions",
            }
            if API类型 == "其他":
                请求地址 = 请求地址 or ""
            else:
                请求地址 = provider_map.get(API类型, 请求地址)
            if not 请求地址:
                return ("请求地址为空，请填写或选择有效的 API 类型",)
            # image -> base64（保持原始尺寸，使用无损 PNG 编码）
            pil = tensor2pil(image)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            # 提示词
            if 输出语言 == "中文":
                提示词_full = f"{提示词} 请用中文返回描述"
            else:
                提示词_full = f"{提示词} Please return the description in English."

            # 按照示例将图片放到 messages[0].content 中
            image_data_uri = f"data:image/png;base64,{img_b64}"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": 提示词_full},
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                    ],
                }
            ]

            payload = {"model": 模型名称, "stream": False, "messages": messages, "max_tokens": 400, "temperature": float(温度)}
            headers = {
                "Authorization": f"Bearer {API_Key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "ComfyUI-DaNodes/1.0",
            }

            # 固定使用硅基流动的 JSON 风格请求
            try:
                r = requests.post(请求地址, json=payload, headers=headers, timeout=30)
            except Exception as e:
                return (f"API 请求失败（网络/连接错误）: {e}",)

            if not r.ok:
                body = None
                try:
                    body = r.json()
                    body = json.dumps(body, ensure_ascii=False)
                except Exception:
                    body = r.text
                return (f"API 返回错误状态 {r.status_code}: {body}",)

            try:
                result = r.json()
            except Exception as e:
                return (f"无法解析 API 返回为 JSON: {e}; body: {r.text}",)

            # 提取文本回复的帮助函数
            def extract_text_from_content(content):
                if isinstance(content, str):
                    return content
                if isinstance(content, dict):
                    return content.get("text") or content.get("content") or str(content)
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if not isinstance(block, dict):
                            parts.append(str(block))
                            continue
                        btype = block.get("type", "")
                        if btype in ("text", "paragraph"):
                            parts.append(block.get("text", ""))
                        elif btype == "image_url":
                            continue
                        else:
                            parts.append(block.get("text") or block.get("content") or "")
                    return " ".join(p for p in parts if p).strip()
                return str(content)

            caption = ""
            if isinstance(result, dict):
                choices = result.get("choices") or []
                if choices:
                    first = choices[0]
                    if isinstance(first, dict):
                        msg = first.get("message") or first.get("delta") or None
                        if isinstance(msg, dict) and "content" in msg:
                            caption = extract_text_from_content(msg.get("content"))
                        else:
                            if "text" in first and isinstance(first.get("text"), str):
                                caption = first.get("text")
                            else:
                                caption = extract_text_from_content(first)
                    else:
                        caption = str(first)
                else:
                    caption = result.get("caption") or result.get("text") or ""
            else:
                caption = str(result)

            caption = caption.strip()
            if caption == "":
                try:
                    pretty = json.dumps(result, ensure_ascii=False)
                except Exception:
                    pretty = str(result)
                return (f"无法从 API 响应解析出文本。返回 JSON: {pretty}",)

            return (caption,)
        except Exception as e:
            # 捕获整个函数体的未处理异常，避免 ComfyUI 无输出
            return (f"节点内部异常: {e}",)

NODE_CLASS_MAPPINGS = {
    "api_caption": api_caption
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "api_caption": "API 打标（精简版）☀"
}