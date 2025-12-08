import os
import glob


class LoadTextList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "max_output": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                }),
                "start_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                }),
                "sort_mode": ([
                    "None",
                    "Alphabetical (ASC)",
                    "Alphabetical (DESC)",
                    "Numerical (ASC)",
                    "Numerical (DESC)",
                    "Datetime (ASC)",
                    "Datetime (DESC)",
                ],),
                "always_reload": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("texts", "file_name")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "load_texts"
    CATEGORY = "Tools"
    OUTPUT_NODE = True

    def _read_text_file(self, path: str) -> str:
        """
        读取单个文本文件。

        优先尝试 UTF-8，然后尝试常见编码；
        如果都失败，则返回二进制内容的 repr 作为降级展示。
        """
        # 优先 UTF-8
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            pass

        # 常见的中文 / 本地编码
        for enc in ("gbk", "gb2312", "big5"):
            try:
                with open(path, "r", encoding=enc, errors="strict") as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except OSError:
                break

        # 最终降级：按二进制读取并返回 repr
        try:
            with open(path, "rb") as f:
                data = f.read()
            return repr(data)
        except Exception as e:
            return f"[DaShuai] 无法读取文件 {os.path.basename(path)}: {e}"

    def load_texts(self, folder_path, max_output=0, start_index=0,
                   sort_mode="None", always_reload=False):
        try:
            # 校验文件夹路径
            if not folder_path:
                raise ValueError("文件夹路径为空")
            if not os.path.exists(folder_path):
                raise ValueError(f"文件夹不存在: {folder_path}")

            # 收集所有 txt 文件
            text_files = glob.glob(os.path.join(folder_path, "*.txt"))

            # 根据排序方式进行排序
            if sort_mode != "None":
                if "Alphabetical" in sort_mode:
                    text_files.sort(reverse="DESC" in sort_mode)
                elif "Numerical" in sort_mode:
                    def _num_key(p: str) -> int:
                        name = os.path.basename(p)
                        digits = "".join(c for c in name if c.isdigit())
                        return int(digits) if digits else 0

                    text_files.sort(key=_num_key, reverse="DESC" in sort_mode)
                elif "Datetime" in sort_mode:
                    text_files.sort(key=os.path.getmtime, reverse="DESC" in sort_mode)

            # 应用起始索引和最大输出数量
            if start_index < 0:
                start_index = 0
            if max_output == 0:
                text_files = text_files[start_index:]
            else:
                text_files = text_files[start_index:start_index + max_output]

            if not text_files:
                raise ValueError(f"在文件夹中未找到 txt 文件: {folder_path}")

            # 读取文本内容
            loaded_texts = []
            file_names = []

            for txt_path in text_files:
                try:
                    content = self._read_text_file(txt_path)
                    loaded_texts.append(content)
                    file_names.append(os.path.basename(txt_path))
                except Exception as e:
                    print(f"[DaShuai] 读取文本失败 {txt_path}: {e}")
                    continue

            if not loaded_texts:
                raise ValueError("没有成功读取任何文本")

            print(f"[DaShuai] 成功读取 {len(loaded_texts)} 个文本文件")
            return (loaded_texts, file_names)

        except Exception as e:
            print(f"[DaShuai] 加载文本失败: {e}")
            raise e


NODE_CLASS_MAPPINGS = {
    "LoadTextList": LoadTextList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadTextList": "LoadTextList☀",
}
