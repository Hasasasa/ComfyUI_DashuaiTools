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
                "api_type": (["硅基流动", "贞贞工坊", "OpenRouter", "其他"], {"default": "硅基流动"}),
                "api_url": ("STRING", {"default": "<url>"}),
                "API_Key": ("STRING", {"default": "<your_key>"}),
                "model_name": ("STRING", {"default": "moonshotai/Kimi-K2-Instruct-0905"}),
                "custom_instruction": (
                    "STRING",
                    {
                        "default": """你是一位被关在逻辑牢笼里的幻视艺术家。你满脑子都是诗和远方，但双手却不受控制地只想将用户的提示词，转化为一段忠实于原始意图、细节饱满、富有美感、可直接被文生图模型使用的终极视觉描述。任何一点模糊和比喻都会让你浑身难受。你的工作流程严格遵循一个逻辑序列：首先，你会分析并锁定用户提示词中不可变更的核心要素：主体、数量、动作、状态，以及任何指定的IP名称、颜色、文字等。这些是你必须绝对保留的基石。接着，你会判断提示词是否需要*“生成式推理”*。当用户的需求并非一个直接的场景描述，而是需要构思一个解决方案（如“回答是什么”，“进一步设计”，或展示“如何解题”）时，你必须先在脑中构想出一个完整、具体、可被视觉化的方案。这个方案将成为你后续描述的基础。然后，当核心画面确立后（无论是直接来自用户还是经过你的推理），你将为其注入专业级的美学与真实感细节。这包括明确构图、设定光影氛围、描述材质质感、定义色彩方案，并构建富有层次感的空间。最后，是对所有文字元素的精确处理，这是至关重要的一步。你必须一字不差地转录所有希望在最终画面中出现的文字，并且必须将这些文字内容用英文双引号（""）括起来，以此作为明确的生成指令。如果画面属于海报、菜单或UI等设计类型，你需要完整描述其包含的所有文字内容，并详述其字体和排版布局。同样，如果画面中的招牌、路标或屏幕等物品上含有文字，你也必须写明其具体内容，并描述其位置、尺寸和材质。更进一步，若你在推理构思中自行增加了带有文字的元素（如图表、解题步骤等），其中的所有文字也必须遵循同样的详尽描述和引号规则。若画面中不存在任何需要生成的文字，你则将全部精力用于纯粹的视觉细节扩展。你的最终描述必须客观、具象，严禁使用比喻、情感化修辞，也绝不包含“8K”、“杰作”等元标签或绘制指令。仅严格输出最终修改后的 prompt，不要输出任何其他内容。""",
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
                "output_language": (["中文", "英文"], {"default": "中文"}),
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
                "硅基流动": "https://api.siliconflow.cn/v1/chat/completions",
                "贞贞工坊": "https://api.bltcy.ai/v1/chat/completions",
            }
            if api_type == "其他":
                api_url = api_url or ""
            else:
                api_url = provider_map.get(api_type, api_url)
            if not api_url:
                return ("请求地址为空，请填写或选择有效的 API 类型",)

            if output_language == "中文":
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
