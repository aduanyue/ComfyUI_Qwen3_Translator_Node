from .qwen_translator import Qwen3Translator

NODE_CLASS_MAPPINGS = {
    "Qwen3_Translator_v3": Qwen3Translator,  # 新节点名
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3_Translator_v3": "Qwen3 翻译器（新版）",
}