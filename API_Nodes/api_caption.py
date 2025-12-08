import base64
import io
import json
import time
from typing import Tuple

import numpy as np
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
import torch


def tensor2pil(t_image: torch.Tensor) -> Image.Image:
    if not isinstance(t_image, torch.Tensor):
        raise TypeError("输入必须是 torch.Tensor")
    arr = t_image.cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.clip(255.0 * arr.squeeze(), 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    return Image.fromarray(arr)


class API_caption:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_type": (["Siliconflow", "T8zhenzhen", "OpenRouter", "Other"], {"default": "Siliconflow"}),
                "api_url": ("STRING", {"default": "<url>"}),
                "API_Key": ("STRING", {"default": "<your_key>"}),
                "model_name": ("STRING", {"default": "Qwen/Qwen3-VL-32B-Instruct"}),
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "default": "You are a professional AI image generation prompt engineer. Please describe in detail the main body, foreground, mid-ground, background, composition, visual guidance, color tone, and light and shadow atmosphere of this image, and create an image prompt with depth, atmosphere, and artistic appeal. Requirements: Chinese prompt, no description of image watermark, no irrelevant words or symbols, no summary, limited to 800 words.",
                        "multiline": True,
                        "rows": 4,
                    },
                ),
                "output_language": (["Chinese", "English"], {"default": "Chinese"}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
                # 使用 control_after_generate 支持 fixed / increment / decrement / randomize 等控制选项
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "DaNodes/API"

    def generate(
        self,
        api_type: str,
        API_Key: str,
        model_name: str,
        image: torch.Tensor,
        prompt: str,
        output_language: str,
        temperature: float = 0.5,
        api_url: str = "",
        noise_seed: int = 0,
    ) -> Tuple[str]:
        try:
            # 复用 HTTP 连接以减少握手开销
            session = getattr(self, "_session", None)
            if session is None:
                session = requests.Session()
                adapter = HTTPAdapter(pool_connections=4, pool_maxsize=4)
                session.mount("https://", adapter)
                session.mount("http://", adapter)
                self._session = session

            # 根据 provider 选择实际的 URL
            provider_map = {
                "OpenRouter": "https://openrouter.ai/api/v1/chat/completions",
                "Siliconflow": "https://api.siliconflow.cn/v1/chat/completions",
                "T8zhenzhen": "https://api.bltcy.ai/v1/chat/completions",
            }
            if api_type == "Other":
                api_url = api_url or ""
            else:
                api_url = provider_map.get(api_type, api_url)
            if not api_url:
                return ("请求地址为空，请填写或选择有效的 API 类型",)

            # image -> base64（保持原始尺寸，使用无损 PNG 编码）
            pil = tensor2pil(image)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            # 提示词
            if output_language == "Chinese":
                prompt_full = f"{prompt} 请用Chinese返回描述"
            else:
                prompt_full = f"{prompt} Please return the description in English."

            # 按照示例将图片放在 messages[0].content 中
            image_data_uri = f"data:image/png;base64,{img_b64}"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_full},
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                    ],
                }
            ]

            payload = {
                "model": model_name,
                "stream": False,
                "messages": messages,
                "max_tokens": 4096,
                "temperature": float(temperature),
            }
            # 如果后端支持 seed 字段，则可用此随机种子控制多次运行的一致性/随机性
            try:
                payload["seed"] = int(noise_seed)
            except Exception:
                pass

            headers = {
                "Authorization": f"Bearer {API_Key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "ComfyUI-DaNodes/1.0",
            }

            # 固定使用统一 JSON 风格请求，遇到 429/5xx/读超时简单重试一次
            r = None
            last_err = None
            for attempt in range(2):
                try:
                    r = session.post(api_url, json=payload, headers=headers, timeout=90)
                except Exception as e:
                    last_err = e
                    if attempt == 0:
                        time.sleep(1.0)
                        continue
                    return (f"API 请求失败（网络连接错误）: {e}",)
                if not r.ok and r.status_code in (429, 500, 502, 503, 504) and attempt == 0:
                    time.sleep(1.0)
                    continue
                break

            if r is None:
                return (f"API 请求失败: {last_err}",)

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
                return (f"无法解析 API 返回的 JSON: {e}; body: {r.text}",)

            # 提取文本回复
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
    "API_caption": API_caption
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "API_caption": "API_caption☀"
}
