import os
import base64
import json
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


class Batch_API_caption:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_dir": ("STRING", {"default": ""}),
                "output_dir": ("STRING", {"default": ""}),
                "api_type": (["Siliconflow", "T8zhenzhen", "OpenRouter", "Other"], {"default": "Siliconflow"}),
                "api_url": ("STRING", {"default": "<url>"}),
                "API_Key": ("STRING", {"default": "<your_key>"}),
                "model_name": ("STRING", {"default": "Qwen/Qwen3-VL-32B-Instruct"}),
                "prompt": ("STRING", {"default": "You are a professional AI image generation prompt engineer. Please describe in detail the main body, foreground, mid-ground, background, composition, visual guidance, color tone, and light and shadow atmosphere of this image, and create an image prompt with depth, atmosphere, and artistic appeal. Requirements: Chinese prompt, no description of image watermark, no irrelevant words or symbols, no summary, limited to 800 words.", "multiline": True, "rows": 4}),
                "output_language": (["Chinese", "English"], {"default": "Chinese"}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 258, "min": 125, "max": 4096}),
                "concurrency": ("INT", {"default": 6, "min": 1, "max": 32}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("log", "save_path")
    FUNCTION = "generate"
    CATEGORY = "DaNodes/API"

    @staticmethod
    def _extract_text_from_content(content):
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

    @staticmethod
    def _build_payload(model_name: str, prompt_full: str, img_data_uri: str, temperature: float, max_tokens: int):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_full},
                    {"type": "image_url", "image_url": {"url": img_data_uri}},
                ],
            }
        ]
        return {
            "model": model_name,
            "stream": False,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }

    @staticmethod
    def _post(api_url: str, api_key: str, payload: dict, timeout_sec: float = 90.0, retries: int = 0):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "ComfyUI-DaNodes/1.0",
        }
        attempt = 0
        last_err = None
        result = None
        while attempt <= max(0, int(retries)):
            try:
                r = requests.post(api_url, json=payload, headers=headers, timeout=float(timeout_sec))
            except Exception as e:
                last_err = f"请求失败: {e}"
                attempt += 1
                continue

            if not r.ok:
                try:
                    body = r.json()
                    body = json.dumps(body, ensure_ascii=False)
                except Exception:
                    body = r.text
                last_err = f"HTTP {r.status_code}: {body}"
                attempt += 1
                continue

            try:
                result = r.json()
            except Exception as e:
                last_err = f"解析 JSON 失败: {e}; body: {r.text}"
                attempt += 1
                continue
            # 成功
            break

        if result is None:
            return False, (last_err or "请求失败")

        # 解析文本
        caption = ""
        if isinstance(result, dict):
            choices = result.get("choices") or []
            if choices:
                first = choices[0]
                if isinstance(first, dict):
                    msg = first.get("message") or first.get("delta") or None
                    if isinstance(msg, dict) and "content" in msg:
                        caption = Batch_API_caption._extract_text_from_content(msg.get("content"))
                    else:
                        if "text" in first and isinstance(first.get("text"), str):
                            caption = first.get("text")
                        else:
                            caption = Batch_API_caption._extract_text_from_content(first)
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
            return False, f"无法解析文本，返回 JSON: {pretty}"

        return True, caption

    def generate(self, input_dir: str, output_dir: str, api_type: str, api_url: str, API_Key: str, model_name: str,
                 prompt: str, output_language: str, temperature: float = 0.5, max_tokens: int = 512,
                 concurrency: int = 4) -> Tuple[str, str]:
        # 目录校验
        if not input_dir:
            return ("输入地址未填写", output_dir)
        if not os.path.isdir(input_dir):
            return (f"输入地址不存在: {input_dir}", output_dir)
        # API Key 校验：为空或仍为默认占位时直接报错
        if not API_Key or API_Key.strip() in ("<your_key>", "your_key", "请输入你的 Key"):
            return ("API 密钥未填写，请在节点中填写有效的 API_Key 后再运行", output_dir)
        if not output_dir:
            output_dir = input_dir
        os.makedirs(output_dir, exist_ok=True)

        # 选择 URL
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
            return ("请求地址为空，请填写或选择有效的 API 类型", output_dir)

        # 列出图片
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(exts)]
        if not files:
            return ("输入地址中没有图片文件", output_dir)

        log = {"total": len(files), "succeeded": 0, "failed": 0}
        first_error = None  # type: ignore[assignment]

        def process_one(name: str):
            in_path = os.path.join(input_dir, name)
            out_txt = os.path.join(output_dir, os.path.splitext(name)[0] + ".txt")
            try:
                # 直接嵌入原始文件字节，不做缩放与重新编码
                with open(in_path, 'rb') as fbin:
                    raw = fbin.read()
                img_b64 = base64.b64encode(raw).decode()
                ext = os.path.splitext(in_path)[1].lower()
                mime = "image/jpeg" if ext in (".jpg", ".jpeg") else (
                    "image/png" if ext == ".png" else (
                    "image/webp" if ext == ".webp" else (
                    "image/bmp" if ext == ".bmp" else "application/octet-stream"))
                )
                img_data_uri = f"data:{mime};base64,{img_b64}"

                if output_language == "Chinese":
                    prompt_full = f"{prompt} 请用Chinese返回描述"
                else:
                    prompt_full = f"{prompt} Please return the description in English."

                payload = self._build_payload(model_name, prompt_full, img_data_uri, temperature, max_tokens)
                ok, resp = self._post(api_url, API_Key, payload, timeout_sec=90.0, retries=1)

                if ok:
                    with open(out_txt, "w", encoding="utf-8") as f:
                        f.write(resp)
                    return True, name, None
                else:
                    # 失败时不创建/不写入任何 txt 文件
                    return False, name, resp
            except Exception as e:
                return False, name, str(e)

        workers = max(1, int(concurrency))
        files_sorted = list(files)
        try:
            files_sorted.sort()
        except Exception:
            pass
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(process_one, n): n for n in files_sorted}
            for fut in as_completed(futures):
                ok, name, err = fut.result()
                if ok:
                    log["succeeded"] += 1
                else:
                    log["failed"] += 1
                    if first_error is None:
                        first_error = (err or "unknown")

        if first_error is not None:
            log["error"] = first_error
        return (json.dumps(log, ensure_ascii=False), output_dir)


NODE_CLASS_MAPPINGS = {
    "Batch_API_caption": Batch_API_caption,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Batch_API_caption": "Batch_API_caption☀",
}
