import importlib
import os
import shutil
import sys
import time

WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def load_all_nodes(root_dir):
    success_nodes = []
    failed_nodes = []
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
                            if display_name:
                                display_name = display_name[:-1]
                            success_nodes.append((name, display_name))
                    if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                        NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
                except Exception as e:
                    failed_nodes.append(f"{module_name}: {e}")

    separator = "=" * 64
    print(separator)
    print("🌞🌞🌞 DaShuai Tools 🌞🌞🌞")
    print(f"✅ 已加载 {len(success_nodes)} 个节点:")
    for name, display_name in success_nodes:
        if display_name != name:
            print(f"  🌞 {display_name} /{name}")
        else:
            print(f"  🌞 {name}")
    if failed_nodes:
        print(f"❌ 加载失败 {len(failed_nodes)} 个节点:")
        for name, display_name in failed_nodes:
            if display_name != name:
                print(f"  🥹 {display_name} /{name}")
            else:
                print(f"  🥹 {name}")
    print(separator)


load_all_nodes(os.path.dirname(__file__))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
