# ComfyUI 节点：batch_api_caption（批量版，简洁可用）
import os
import base64
import json
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


class batch_api_caption:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "输入地址": ("STRING", {"default": ""}),
                "输出地址": ("STRING", {"default": ""}),
                "API类型": (["硅基流动", "贞贞工坊", "OpenRouter", "其他"], {"default": "硅基流动"}),
                "请求地址": ("STRING", {"default": "<url>"}),
                "API_Key": ("STRING", {"default": "<your_key>"}),
                "模型名称": ("STRING", {"default": "Qwen/Qwen2.5-VL-32B-Instruct"}),
                "提示词": ("STRING", {"default": "请用自然语言描述该图片。直接返回描述，不要有其他废话。字数控制再50字以内。", "multiline": True, "rows": 4}),
                "输出语言": (["中文", "英文"], {"default": "中文"}),
                "温度": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
                "并发数": ("INT", {"default": 4, "min": 1, "max": 32}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log",)
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
    def _build_payload(model_name: str, prompt_full: str, img_data_uri: str, temperature: float):
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
            "max_tokens": 400,
            "temperature": float(temperature),
        }

    @staticmethod
    def _post(api_url: str, api_key: str, payload: dict, timeout_sec: float = 30.0, retries: int = 0):
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
                        caption = batch_api_caption._extract_text_from_content(msg.get("content"))
                    else:
                        if "text" in first and isinstance(first.get("text"), str):
                            caption = first.get("text")
                        else:
                            caption = batch_api_caption._extract_text_from_content(first)
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

    def generate(self, 输入地址: str, 输出地址: str, API类型: str, 请求地址: str, API_Key: str, 模型名称: str,
                 提示词: str, 输出语言: str, 温度: float = 0.5,
                 并发数: int = 4) -> Tuple[str]:
        # 目录校验
        if not 输入地址:
            return ("输入地址未填写",)
        if not os.path.isdir(输入地址):
            return (f"输入地址不存在: {输入地址}",)
        if not 输出地址:
            输出地址 = 输入地址
        os.makedirs(输出地址, exist_ok=True)

        # 选择 URL
        provider_map = {
            "OpenRouter": "https://openrouter.ai/api/v1/chat/completions",
            "硅基流动": "https://api.siliconflow.cn/v1/chat/completions",
            "贞贞工坊": "https://api.bltcy.ai/v1/chat/completions",
        }
        if API类型 == "其他":
            api_url = 请求地址 or ""
        else:
            api_url = provider_map.get(API类型, 请求地址)
        if not api_url:
            return ("请求地址为空，请填写或选择有效的 API 类型",)

        # 列出图片
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        files = [f for f in os.listdir(输入地址) if f.lower().endswith(exts)]
        if not files:
            return ("输入地址中没有图片文件",)

        log = {"total": len(files), "succeeded": 0, "failed": 0}
        first_error = None  # type: ignore[assignment]

        def process_one(name: str):
            in_path = os.path.join(输入地址, name)
            out_txt = os.path.join(输出地址, os.path.splitext(name)[0] + ".txt")
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

                if 输出语言 == "中文":
                    prompt_full = f"{提示词} 请用中文返回描述"
                else:
                    prompt_full = f"{提示词} Please return the description in English."

                payload = self._build_payload(模型名称, prompt_full, img_data_uri, 温度)
                ok, resp = self._post(api_url, API_Key, payload, timeout_sec=30.0, retries=1)

                if ok:
                    with open(out_txt, "w", encoding="utf-8") as f:
                        f.write(resp)
                    return True, name, None
                else:
                    # 失败时不创建/不写入任何 txt 文件
                    return False, name, resp
            except Exception as e:
                return False, name, str(e)

        workers = max(1, int(并发数))
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
        return (json.dumps(log, ensure_ascii=False),)


NODE_CLASS_MAPPINGS = {
    "batch_api_caption": batch_api_caption,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "batch_api_caption": "API 批量打标（精简版）☀",
}
