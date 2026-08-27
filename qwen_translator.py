import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, __version__ as transformers_version
import folder_paths
import os
import json
from packaging import version
import re

try:
    from transformers import Qwen3VLForConditionalGeneration
    VL_AVAILABLE = True
except ImportError:
    Qwen3VLForConditionalGeneration = None
    VL_AVAILABLE = False


class Qwen3Translator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded_model_path = None
        self.current_model_type = "text"

    @classmethod
    def INPUT_TYPES(cls):
        model_list = cls.get_available_models()
        if not model_list:
            model_list = ["[自定义路径]"]
        
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "source_lang": (["auto", "中文", "English", "日本語", "한국어"], {"default": "auto"}),
                "target_lang": (["中文", "English", "日本語", "한국어"], {"default": "English"}),
                "model_choice": (model_list, {"default": model_list[0] if model_list else ""}),
                "custom_path": ("STRING", {"default": "", "placeholder": "留空则使用下拉选择"}),
                "max_new_tokens": ("INT", {"default": 4096, "min": 64, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "translate"
    CATEGORY = "翻译（新版）"

    @classmethod
    def get_available_models(cls):
        search_dirs = [
            folder_paths.models_dir + "/text_encoders",
            folder_paths.models_dir + "/LLM",
            folder_paths.models_dir + "/prompt_generator",
        ]
        model_names = []
        for base_dir in search_dirs:
            if not os.path.isdir(base_dir):
                continue
            for item in os.listdir(base_dir):
                full_path = os.path.join(base_dir, item)
                if os.path.isdir(full_path):
                    config_path = os.path.join(full_path, "config.json")
                    if os.path.isfile(config_path):
                        try:
                            with open(config_path, 'r', encoding='utf-8') as f:
                                config = json.load(f)
                            if config.get("model_type"):
                                model_names.append(item)
                        except:
                            model_names.append(item)
        model_names = sorted(list(set(model_names)))
        return model_names

    def get_model_path(self, model_choice, custom_path):
        if custom_path and os.path.isdir(custom_path):
            return custom_path
        elif model_choice and model_choice != "[自定义路径]":
            search_dirs = [
                folder_paths.models_dir + "/text_encoders",
                folder_paths.models_dir + "/LLM",
                folder_paths.models_dir + "/prompt_generator",
            ]
            for base_dir in search_dirs:
                full_path = os.path.join(base_dir, model_choice)
                if os.path.isdir(full_path):
                    return full_path
            if os.path.isdir(model_choice):
                return model_choice
        return ""

    def detect_model_type(self, model_path):
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                model_type = config.get("model_type", "")
                return model_type.lower()
            except:
                pass
        return "text"

    def is_model_supported(self, model_type):
        unsupported = ["qwen3_5", "qwen3.5"]
        if model_type in unsupported:
            try:
                if version.parse(transformers_version) < version.parse("4.58.0"):
                    return False
            except:
                return False
        return True

    def translate(self, text, source_lang, target_lang, model_choice, custom_path, max_new_tokens, temperature):
        model_path = self.get_model_path(model_choice, custom_path)
        if not model_path:
            raise ValueError("❌ 未找到有效的本地模型路径。请确认模型文件夹存在且包含 config.json。")

        model_type = self.detect_model_type(model_path)
        print(f"[Qwen3 Translator] 检测到模型类型: {model_type}")

        if self.loaded_model_path != model_path:
            print(f"[Qwen3 Translator] 📥 加载本地模型: {model_path}")

            if not self.is_model_supported(model_type):
                if model_type in ["qwen3_5", "qwen3.5"]:
                    raise ValueError(
                        f"❌ 当前 transformers 版本 ({transformers_version}) 不支持模型类型 '{model_type}'。\n"
                        "解决方案（二选一）：\n"
                        "  1. 升级 transformers 到 4.58.0 或更高版本：pip install --upgrade transformers\n"
                        "     注意：升级后可能需要设置环境变量 TRANSFORMERS_DISABLE_FINEGRAINED_FP8=1 以避免其他问题。\n"
                        "  2. 更换为兼容的模型，例如 Qwen3-4B 或 Qwen3-8B（纯文本版本），或继续使用 Qwen3-VL-4B-Instruct-FP8。\n"
                        "下载兼容模型示例：\n"
                        "  huggingface-cli download Qwen/Qwen3-4B --local-dir ComfyUI/models/text_encoders/Qwen3-4B"
                    )
                else:
                    raise ValueError(f"❌ 未知或不支持的模型类型: {model_type}")

            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            if "vl" in model_type or "vision" in model_type:
                if VL_AVAILABLE and Qwen3VLForConditionalGeneration is not None:
                    print("[Qwen3 Translator] 使用 VL 专用加载器")
                    self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                else:
                    raise ValueError(
                        f"❌ 当前 transformers 版本 ({transformers_version}) 不支持加载视觉语言模型 (VL)。\n"
                        "请使用纯文本模型（如 Qwen3-4B 或 Qwen3-8B）进行翻译，\n"
                        "或者升级 transformers 到 5.x 版本并设置环境变量 TRANSFORMERS_DISABLE_FINEGRAINED_FP8=1。"
                    )
            else:
                print("[Qwen3 Translator] 使用标准 CausalLM 加载器")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            self.model.eval()
            self.loaded_model_path = model_path
            self.current_model_type = model_type
        else:
            print("[Qwen3 Translator] ✅ 使用已加载的模型")
            model_type = self.current_model_type

        lang_map = {
            "中文": "Chinese",
            "English": "English",
            "日本語": "Japanese",
            "한국어": "Korean"
        }
        src = "auto" if source_lang == "auto" else lang_map[source_lang]
        tgt = lang_map[target_lang]

        # 构建翻译指令（针对不同模型类型优化）
        if "vl" in model_type or "vision" in model_type:
            prompt = f"Translate the following text into {tgt}. Only output the translation, no other text:\n{text}"
        else:
            prompt = f"Translate the following English text into {tgt}. Output only the translation:\n{text}"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🔵 完整解码结果: {result}")

        # --- 提取方括号内的翻译 ---
        matches = re.findall(r'\[([\s\S]*?)\]', result)
        if len(matches) >= 2:
            translation = matches[1].strip()
            if any('\u4e00' <= ch <= '\u9fff' for ch in translation):
                print(f"🔵 正则提取翻译成功")
                result = translation
            else:
                print("⚠️ 第二个括号内不是中文，尝试其他提取方法")
        else:
            print("⚠️ 未找到足够的方括号对，使用回退提取")

        # 回退逻辑：如果提取结果不包含中文，则提取包含中文的最后几行
        if not any('\u4e00' <= ch <= '\u9fff' for ch in result):
            print("⚠️ 使用回退提取逻辑")
            lines = result.strip().split('\n')
            for i in range(min(10, len(lines)), 0, -1):
                candidate = "\n".join(lines[-i:]).strip()
                if any('\u4e00' <= ch <= '\u9fff' for ch in candidate):
                    result = candidate
                    print(f"🔵 回退提取包含中文的最后 {i} 行作为结果")
                    break
            else:
                if lines:
                    result = lines[-1].strip()
                    print("🔵 回退提取最后一行作为结果")

        # --- JSON 解码转义字符（如 \" 和 \n）---
        try:
            # 将结果作为 JSON 字符串解码（保留实际的换行符和引号）
            decoded = json.loads(f'"{result}"')
            if isinstance(decoded, str):
                result = decoded
                print("🔵 成功解码 JSON 转义字符")
        except Exception as e:
            print(f"⚠️ JSON 解码失败，使用原始结果: {e}")

        # --- 最终清理：去除换行符和多余空格 ---
        # 将所有换行符替换为空格
        result = result.replace('\n', ' ')
        # 将多个连续空格压缩为单个空格
        result = ' '.join(result.split())
        # 去除首尾空格
        result = result.strip()

        print(f"🔵 最终返回的结果: {result[:200]}...")
        return (result,)