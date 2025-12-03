import importlib
import os
import shutil
import sys
import time
import json

WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _load_locale_display_names():
    """Load zh locale display names for logging purposes."""
    locale_map = {}
    try:
        base_dir = os.path.dirname(__file__)
        locale_path = os.path.join(base_dir, "locales", "zh", "nodeDefs.json")
        if os.path.exists(locale_path):
            with open(locale_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key, cfg in data.items():
                disp = cfg.get("display_name")
                if isinstance(disp, str) and disp:
                    # 去掉末尾装饰符（如 ☀）以及多余空格
                    clean = disp.rstrip()
                    if clean.endswith("☀"):
                        clean = clean[:-1].rstrip()
                    locale_map[key] = clean
    except Exception:
        # Logging is best-effort only; ignore locale errors
        pass
    return locale_map


def load_all_nodes(root_dir):
    success_nodes = []
    failed_nodes = []

    zh_display_map = _load_locale_display_names()

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py") and filename != "__init__.py":
                module_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(module_path, root_dir)
                module_name = rel_path[:-3].replace(os.sep, ".")
                try:
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, "NODE_CLASS_MAPPINGS"):
                        NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                        display_map = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}) or {}
                        for name in module.NODE_CLASS_MAPPINGS.keys():
                            display_name = display_map.get(name, name)
                            # 去掉末尾的 ☀，用于英文内部名展示
                            if display_name and display_name.endswith("☀"):
                                display_name = display_name[:-1]
                            zh_name = zh_display_map.get(name, "")
                            success_nodes.append((name, display_name, zh_name))
                    if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                        NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
                except Exception as e:
                    # store as tuple (module_name, error_message) so callers can unpack
                    failed_nodes.append((module_name, str(e)))

    separator = "=" * 64
    print(separator)
    print("🌞🌞🌞 DaShuai Tools 🌞🌞🌞")
    print(f"✅ 已加载 {len(success_nodes)} 个节点:")

    # 对齐英文列，让中文描述排版更整齐
    max_eng_len = 0
    for name, _, _ in success_nodes:
        if len(str(name)) > max_eng_len:
            max_eng_len = len(str(name))

    for name, display_name, zh_name in success_nodes:
        eng = name
        cn = zh_name if zh_name else ""
        eng_padded = eng.ljust(max_eng_len)
        if cn and cn != eng:
            # 例如：🌞 api_caption      🌞 API 打标
            print(f"  🌞 {eng_padded}  🌞 {cn}")
        else:
            print(f"  🌞 {eng}")
    if failed_nodes:
        print(f"❌ 加载失败 {len(failed_nodes)} 个节点:")
        for name, err in failed_nodes:
            print(f"  🥹 {name}: {err}")
    print(separator)


load_all_nodes(os.path.dirname(__file__))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
