from .train import LoraTraininginComfySDXL, LoraTraininginComfySDXLJSON, TensorboardAccess
NODE_CLASS_MAPPINGS = {
    "Lora Training in Comfy (SDXL)": LoraTraininginComfySDXL,
    "Lora Training in Comfy (SDXL JSON)": LoraTraininginComfySDXLJSON,
    "Tensorboard Access": TensorboardAccess
}
NODE_DISPLAY_NAME_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']