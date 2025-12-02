import torch
import comfy.utils
import comfy.sd
import folder_paths
from nodes import LoraLoader
#代码来自于https://github.com/PGCRT/CRT-Nodes/blob/main/py/LoraLoaderZImage.py。
# 由于该插件包内容过多，故直接将这个节点的代码拿出来用了，以支持Z-Image的LoRA加载功能。
class LoraLoaderZImage(LoraLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The Z-Image diffusion model."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the Z-Image LoRA."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_zimage_lora"
    CATEGORY = "ZImage/Loaders"

    loaded_lora = None

    def load_zimage_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        
        # Cache check
        if LoraLoaderZImage.loaded_lora is not None:
            if LoraLoaderZImage.loaded_lora[0] == lora_path:
                lora = LoraLoaderZImage.loaded_lora[1]
            else:
                LoraLoaderZImage.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            
            new_lora = {}
            qkv_groups = {}

            for k, v in lora.items():
                new_k = k

                # Fix output projection
                if ".attention.to_out.0." in new_k:
                    new_k = new_k.replace(".attention.to_out.0.", ".attention.out.")
                    new_lora[new_k] = v
                    continue

                # Collect QKV
                if ".attention.to_" in new_k:
                    parts = new_k.split(".attention.to_")
                    base_prefix = parts[0] + ".attention"
                    remainder = parts[1]

                    qkv_type = remainder[0]
                    suffix = remainder[2:]

                    if base_prefix not in qkv_groups:
                        qkv_groups[base_prefix] = {'q': {}, 'k': {}, 'v': {}}

                    qkv_groups[base_prefix][qkv_type][suffix] = v
                    continue

                # Other layers pass-through
                new_lora[new_k] = v

            # Fuse QKV
            for base_key, group in qkv_groups.items():
                ak_a = "lora_A.weight"
                ak_b = "lora_B.weight"

                # A weights
                if ak_a in group['q'] and ak_a in group['k'] and ak_a in group['v']:
                    fused_A = torch.cat([
                        group['q'][ak_a],
                        group['k'][ak_a],
                        group['v'][ak_a],
                    ], dim=0)
                    new_lora[f"{base_key}.qkv.lora_A.weight"] = fused_A

                # B weights
                if ak_b in group['q'] and ak_b in group['k'] and ak_b in group['v']:
                    q_b = group['q'][ak_b]
                    k_b = group['k'][ak_b]
                    v_b = group['v'][ak_b]

                    out_dim, rank = q_b.shape
                    fused_B = torch.zeros((out_dim * 3, rank * 3), dtype=q_b.dtype, device=q_b.device)

                    fused_B[0:out_dim, 0:rank] = q_b
                    fused_B[out_dim:2*out_dim, rank:2*rank] = k_b
                    fused_B[2*out_dim:3*out_dim, 2*rank:3*rank] = v_b

                    new_lora[f"{base_key}.qkv.lora_B.weight"] = fused_B

                # Alpha
                if "lora_alpha" in group['q']:
                    new_lora[f"{base_key}.qkv.lora_alpha"] = group['q']["lora_alpha"]

            lora = new_lora
            LoraLoaderZImage.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip
        )
        return (model_lora, clip_lora)


NODE_CLASS_MAPPINGS = {
    "LoraLoaderZImage": LoraLoaderZImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraLoaderZImage": "Z-Image LoRA Loader☀"
}
