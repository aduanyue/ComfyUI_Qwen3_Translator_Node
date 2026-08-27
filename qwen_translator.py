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
                "max_new_tokens": ("INT", {"default": 16384, "min": 1024, "max": 32768}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
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

    # ---------- 检查模型文件完整性 ----------
    def verify_model_files(self, model_path):
        index_files = ["model.safetensors.index.json", "pytorch_model.bin.index.json"]
        index_file = None
        for fname in index_files:
            full = os.path.join(model_path, fname)
            if os.path.isfile(full):
                index_file = full
                break

        if index_file is not None:
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    idx_data = json.load(f)
                weight_map = idx_data.get("weight_map", {})
                if weight_map:
                    shard_files = set(weight_map.values())
                else:
                    checkpoints = idx_data.get("checkpoints", [])
                    shard_files = set(checkpoints) if checkpoints else set()
                if not shard_files:
                    return True, []
            except Exception as e:
                print(f"[Qwen3 Translator] 警告：无法解析索引文件 {index_file}，跳过完整性检查。错误：{e}")
                return True, []
        else:
            single_files = [f for f in os.listdir(model_path) 
                           if f.endswith(('.bin', '.safetensors')) and 
                              not f.endswith('.index.json')]
            if single_files:
                return True, []
            else:
                return False, ["未找到任何权重文件（*.bin 或 *.safetensors）"]

        missing = []
        for shard in shard_files:
            shard_path = os.path.join(model_path, shard)
            if not os.path.isfile(shard_path):
                missing.append(shard)
        if missing:
            return False, missing
        return True, []
    # -----------------------------------------

    def translate(self, text, source_lang, target_lang, model_choice, custom_path, max_new_tokens, temperature):
        if not text or not text.strip():
            return ("",)

        model_path = self.get_model_path(model_choice, custom_path)
        if not model_path:
            raise ValueError("❌ 未找到有效的本地模型路径。请确认模型文件夹存在且包含 config.json。")

        # 模型文件完整性检查
        ok, missing_files = self.verify_model_files(model_path)
        if not ok:
            if missing_files:
                missing_list = "\n  - ".join(missing_files)
                raise ValueError(
                    f"❌ 模型目录中缺少以下权重文件：\n  - {missing_list}\n"
                    "请重新下载完整模型文件，确保所有分片文件都存在。\n"
                    f"下载命令示例： huggingface-cli download <模型仓库名> --local-dir {model_path}"
                )
            else:
                raise ValueError(
                    f"❌ 模型目录 {model_path} 中未找到任何权重文件（*.bin 或 *.safetensors）。\n"
                    "请确认模型已正确下载。"
                )

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
                        "  2. 更换为兼容的模型，例如 Qwen3-4B 或 Qwen3-8B（纯文本版本）。\n"
                        "下载兼容模型示例：\n"
                        "  huggingface-cli download Qwen/Qwen3-4B --local-dir ComfyUI/models/text_encoders/Qwen3-4B"
                    )
                else:
                    raise ValueError(f"❌ 未知或不支持的模型类型: {model_type}")

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            except Exception as e:
                raise ValueError(f"❌ 模型加载失败，请检查模型文件是否完整：{e}")

            try:
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
            except FileNotFoundError as fnfe:
                raise ValueError(
                    f"❌ 加载模型时缺少文件：{fnfe.filename if hasattr(fnfe, 'filename') else str(fnfe)}\n"
                    "这通常表示模型下载不完整。请重新下载完整模型文件。"
                ) from fnfe
            except Exception as e:
                raise ValueError(f"❌ 加载模型时发生错误：{e}") from e

            self.model.eval()
            self.loaded_model_path = model_path
            self.current_model_type = model_type
        else:
            print("[Qwen3 Translator] ✅ 使用已加载的模型")
            model_type = self.current_model_type

        # 语言映射
        lang_map = {
            "中文": "Chinese",
            "English": "English",
            "日本語": "Japanese",
            "한국어": "Korean"
        }
        tgt = lang_map[target_lang]

        # ----- 自动检测源语言（仅用于日志） -----
        if source_lang == "auto":
            chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
            japanese_chars = re.findall(r'[\u3040-\u30ff\u4e00-\u9fff]', text)
            korean_chars = re.findall(r'[\uac00-\ud7af]', text)
            if len(chinese_chars) > len(text) * 0.1:
                src_display = "Chinese"
            elif len(japanese_chars) > len(text) * 0.05:
                src_display = "Japanese"
            elif len(korean_chars) > len(text) * 0.05:
                src_display = "Korean"
            else:
                src_display = "English"
            print(f"[Qwen3 Translator] 自动检测源语言（仅日志）: {src_display}")
            src_specified = False
        else:
            src = lang_map[source_lang]
            src_specified = True
            print(f"[Qwen3 Translator] 用户指定源语言: {src}")

        # ----- 构建 prompt（手动使用 Qwen 官方格式） -----
        # 为了最大兼容性，直接使用原始文本指令，不依赖 apply_chat_template，避免回显问题。
        if src_specified:
            instruction = f"Translate the following text from {src} into {tgt}. Output ONLY the translation, without any additional comments or labels. Preserve all details and formatting.\n\n{text}"
        else:
            instruction = f"Translate the following text into {tgt}. Output ONLY the translation, without any additional comments or labels. Preserve all details and formatting.\n\n{text}"

        # 构建符合 Qwen 聊天模板的格式（适用于 Qwen2/2.5/3 系列）
        # 注意：如果模型是 Qwen3，可能使用 <|im_start|> 和 <|im_end|>。
        # 我们优先尝试 apply_chat_template，如果失败则手工构建。
        try:
            messages = [
                {"role": "system", "content": "You are a helpful multilingual translation assistant."},
                {"role": "user", "content": instruction}
            ]
            # 尝试使用 tokenizer 的 chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            print("[Qwen3 Translator] 使用 tokenizer.apply_chat_template 构建 prompt")
        except Exception as e:
            print(f"[Qwen3 Translator] apply_chat_template 失败，使用手工构建: {e}")
            # 手工构建（适用于常见 Qwen 模型）
            # 注意：需要知道模型的特殊 token，但一般可以用通用模板
            # 这里采用更保守的 simple 方式：直接拼接指令
            prompt = instruction

        # 打印完整 prompt（调试用，截断避免日志过大）
        print(f"[Qwen3 Translator] 生成 prompt 长度: {len(prompt)} 字符")
        if len(prompt) > 500:
            print(f"[Qwen3 Translator] Prompt 预览: {prompt[:200]} ... {prompt[-200:]}")
        else:
            print(f"[Qwen3 Translator] Prompt: {prompt}")

        # ----- 生成 -----
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # 设置生成参数
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else None,
            "top_p": 0.9 if temperature > 0 else None,
            "pad_token_id": self.tokenizer.eos_token_id,  # 避免 padding 警告
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        # 移除 None 值
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )

        # 解码时，从 input_ids 长度之后开始，只取生成部分
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        if generated_tokens.shape[0] == 0:
            # 如果生成为空，则整个解码
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("⚠️ 模型未生成任何新 token，输出整个序列")
        else:
            full_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(f"✅ 模型生成了 {generated_tokens.shape[0]} 个新 token")

        print(f"🔵 完整解码结果（前200字符）: {full_output[:200]}...")

        # ---------- 提取翻译 ----------
        raw_translation = full_output.strip()
        # 如果 raw_translation 以 prompt 开头，尝试剥离 prompt（但通常我们只解码生成部分，不会包含 prompt）
        # 但有些模型可能仍会输出 prompt，做安全处理
        if prompt in raw_translation:
            raw_translation = raw_translation.split(prompt, 1)[-1].strip()

        # 去除常见前缀
        for prefix in ["Translation:", "Translated text:", "Output:", "Result:"]:
            if raw_translation.startswith(prefix):
                raw_translation = raw_translation[len(prefix):].strip()
                break

        # 尝试解析 JSON 数组
        bracket_match = re.search(r'(\[[\s\S]*\])', raw_translation)
        if bracket_match:
            try:
                parsed = json.loads(bracket_match.group(1))
                if isinstance(parsed, list) and parsed:
                    raw_translation = str(parsed[-1]).strip()
                    print("🔵 从 JSON 数组提取翻译")
                elif isinstance(parsed, str):
                    raw_translation = parsed.strip()
            except:
                pass

        # 对中文目标，若结果不含中文，尝试提取含中文的行
        if target_lang == "中文":
            if not re.search(r'[\u4e00-\u9fff]', raw_translation):
                lines = raw_translation.split('\n')
                chinese_lines = [line for line in lines if re.search(r'[\u4e00-\u9fff]', line)]
                if chinese_lines:
                    raw_translation = '\n'.join(chinese_lines)
                    print("🔵 提取含中文的行")

        final_translation = re.sub(r'\n\s*\n', '\n', raw_translation).strip()
        print(f"🔵 最终返回结果长度: {len(final_translation)} 字符")
        print(f"🔵 最终结果预览: {final_translation[:200] if final_translation else '(空)'}...")
        return (final_translation,)