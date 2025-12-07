import json
import time
from typing import Tuple

import requests
from requests.adapters import HTTPAdapter


class API_PromptHelper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_type": (["Siliconflow", "T8zhenzhen", "OpenRouter", "Other"], {"default": "Siliconflow"}),
                "api_url": ("STRING", {"default": "<url>"}),
                "API_Key": ("STRING", {"default": "<your_key>"}),
                "model_name": ("STRING", {"default": "moonshotai/Kimi-K2-Instruct-0905"}),
                "custom_instruction": (
                    "STRING",
                    {
                        "default": "You are a visionary artist trapped in a logical cage. Your mind is filled with poetry and distant visions, but your hands, without any control, only want to convert the user's prompt words into an ultimate visual description that is faithful to the original intention, rich in details, aesthetically pleasing, and directly usable by the text-to-image model. Any ambiguity or metaphor will make you feel uncomfortable. Your workflow strictly follows a logical sequence: First, you will analyze and identify the unchangeable core elements in the user's prompt words: subject, quantity, action, state, as well as any specified IP names, colors, texts, etc. These are the fundamental elements that you must absolutely preserve. Then, you will determine if the prompt requires \"generative reasoning\". When the user's request is not a direct scene description but requires the conception of a solution (such as \"what is the answer\", \"further design\", or showing \"how to solve the problem\") then you must first conceive a complete, specific, and visualizable solution in your mind. This solution will be the basis for your subsequent description. Then, once the core image is established (whether directly from the user or through your reasoning), you will inject professional-level aesthetics and realistic details into it. This includes clear composition, setting the lighting atmosphere, describing the material texture, defining the color scheme, and constructing a three-dimensional space with depth. Finally, the precise processing of all text elements is a crucial step. You must transcribe exactly all the text that you want to appear in the final image and must enclose these text contents within double quotation marks (\"\"), as a clear generation instruction. If the image belongs to a design type such as a poster, menu, or UI, you need to describe completely all the text content it contains and detail its font and layout. Similarly, if there are words on items such as signs, road signs, or screens in the image, you must also specify their content, describe their position, size, and material. Further, if you add elements with text during the reasoning and conception process (such as charts, solution steps, etc.), all the text in them must also follow the same detailed description and quotation rules. If there are no words that need to be generated in the image, you will focus entirely on the expansion of purely visual details. Your final description must be objective and concrete. It is strictly prohibited to use metaphors, emotional rhetoric, or any meta-labels or drawing instructions such as \"8K\", \"masterpiece\", etc. Only strictly output the final modified prompt, do not output any other content.",
                        "multiline": True,
                        "rows": 3,
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
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
        custom_instruction: str,
        prompt: str,
        output_language: str,
        temperature: float = 0.5,
        api_url: str = "",
        noise_seed: int = 0,
    ) -> Tuple[str]:
        try:
            session = getattr(self, "_session", None)
            if session is None:
                session = requests.Session()
                adapter = HTTPAdapter(pool_connections=4, pool_maxsize=4)
                session.mount("https://", adapter)
                session.mount("http://", adapter)
                self._session = session

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

            if output_language == "Chinese":
                lang_hint = "请用中文返回优化后的提示词。"
            else:
                lang_hint = "Please return the optimized prompt in English."

            system_prompt = custom_instruction.strip()

            combined_text = f"{system_prompt}\n\n用户输入 prompt: {prompt}\n\n{lang_hint}"

            messages = [
                {"role": "user", "content": combined_text},
            ]

            payload = {
                "model": model_name,
                "stream": False,
                "messages": messages,
                "max_tokens": 800,
                "temperature": float(temperature),
            }
            # 如果后端支持 seed 字段，则可用此随机种子控制多次运行时的一致性/随机性
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
                        parts.append(block.get("text") or block.get("content") or "")
                    return " ".join(p for p in parts if p).strip()
                return str(content)

            text = ""
            if isinstance(result, dict):
                choices = result.get("choices") or []
                if choices:
                    first = choices[0]
                    if isinstance(first, dict):
                        msg = first.get("message") or first.get("delta") or None
                        if isinstance(msg, dict) and "content" in msg:
                            text = extract_text_from_content(msg.get("content"))
                        else:
                            if "text" in first and isinstance(first.get("text"), str):
                                text = first.get("text")
                            else:
                                text = extract_text_from_content(first)
                    else:
                        text = str(first)
                else:
                    text = result.get("text") or ""
            else:
                text = str(result)

            text = text.strip()
            if text == "":
                try:
                    pretty = json.dumps(result, ensure_ascii=False)
                except Exception:
                    pretty = str(result)
                return (f"无法在 API 响应中解析出文本。返回 JSON: {pretty}",)

            return (text,)
        except Exception as e:
            return (f"节点内部异常: {e}",)


NODE_CLASS_MAPPINGS = {
    "API_PromptHelper": API_PromptHelper,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "API_PromptHelper": "API_PromptHelper☀",
}
