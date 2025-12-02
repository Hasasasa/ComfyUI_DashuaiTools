import torch
import comfy.utils
import comfy.sd
import folder_paths
from nodes import LoraLoader

#代码参考了：https://github.com/PGCRT/CRT-Nodes/blob/main/py/LoraLoaderZImage.py
#由GPT结合官方LoRA Model-Only Loader节点进行改写和优化。

# ---------------------------------------------------------------------
#   Z-Image LoRA Model-Only Loader
# ---------------------------------------------------------------------
class ZImageLoraModelOnly(LoraLoader):

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"), ),
                "strength_model": ("FLOAT", {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_zimage_lora_model_only"
    CATEGORY = "loaders"
    DESCRIPTION = "Loads Z-Image LoRA (QKV fused) and applies it to MODEL only (no CLIP)."

    # -------------------------------------------------------
    #  Z-IMAGE QKV FUSION
    # -------------------------------------------------------
    def _convert_zimage_lora(self, lora_dict):
        new_lora = {}
        qkv_groups = {}

        for k, v in lora_dict.items():
            new_k = k

            # 修正 to_out.0 → out
            if ".attention.to_out.0." in new_k:
                new_k = new_k.replace(".attention.to_out.0.", ".attention.out.")
                new_lora[new_k] = v
                continue

            # Q/K/V 三分结构
            if ".attention.to_" in new_k:
                parts = new_k.split(".attention.to_")
                base_prefix = parts[0] + ".attention"
                remainder = parts[1]   # 例如：q.lora_A.weight

                qkv_type = remainder[0]   # q / k / v
                suffix = remainder[2:]    # lora_A.weight

                if base_prefix not in qkv_groups:
                    qkv_groups[base_prefix] = {"q": {}, "k": {}, "v": {}}

                qkv_groups[base_prefix][qkv_type][suffix] = v
                continue

            # 其他权重直接 passthrough
            new_lora[new_k] = v

        # ---- 开始 Fusion ----
        for base_key, group in qkv_groups.items():

            # A 权重
            ak = "lora_A.weight"
            if ak in group["q"] and ak in group["k"] and ak in group["v"]:
                fused_A = torch.cat(
                    [group["q"][ak], group["k"][ak], group["v"][ak]], dim=0
                )
                new_lora[f"{base_key}.qkv.lora_A.weight"] = fused_A

            # B 权重
            bk = "lora_B.weight"
            if bk in group["q"] and bk in group["k"] and bk in group["v"]:
                q_b = group["q"][bk]
                k_b = group["k"][bk]
                v_b = group["v"][bk]

                out_dim, rank = q_b.shape
                fused_B = torch.zeros((out_dim * 3, rank * 3),
                                      dtype=q_b.dtype, device=q_b.device)

                fused_B[0:out_dim, 0:rank] = q_b
                fused_B[out_dim:2*out_dim, rank:2*rank] = k_b
                fused_B[2*out_dim:3*out_dim, 2*rank:3*rank] = v_b

                new_lora[f"{base_key}.qkv.lora_B.weight"] = fused_B

            # alpha
            if "lora_alpha" in group["q"]:
                new_lora[f"{base_key}.qkv.lora_alpha"] = group["q"]["lora_alpha"]

        return new_lora

    # -------------------------------------------------------
    # 主函数
    # -------------------------------------------------------
    def load_zimage_lora_model_only(self, model, lora_name, strength_model):

        # load .safetensors / .pt
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        raw_lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        # 转换为 Z-Image 格式
        fixed_lora = self._convert_zimage_lora(raw_lora)

        # 注入 LoRA（clip=None）
        new_model, _ = comfy.sd.load_lora_for_models(
            model,
            None,
            fixed_lora,
            strength_model,
            0   # clip 强度 = 0
        )

        return (new_model,)


# ---------------------------------------------------------------------
#   节点注册
# ---------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "ZImageLoraModelOnly": ZImageLoraModelOnly,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageLoraModelOnly": "LoRA加载器（Z-Image，仅模型）☀",
}
