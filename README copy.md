
---

# Qwen3 Translator for ComfyUI

一个基于 Qwen3 系列模型的**本地翻译节点**，专为 ComfyUI 设计。  
充分利用你的本地显卡，实现完全离线、高质量、多语言翻译，无需联网，保护数据隐私。

## ✨ 特色

- **完全本地运行**：无需联网，所有数据仅在本地处理。
- **自动扫描本地模型**：自动识别 `ComfyUI/models/text_encoders/` 和 `ComfyUI/models/LLM/` 下的 Qwen 模型，下拉菜单直接选择。
- **支持多种语言**：中文、英文、日文、韩文互译（可扩展更多）。
- **翻译指令优化**：内置专业翻译提示模板，翻译结果更准确自然。
- **模型缓存复用**：同一模型只加载一次，后续使用秒级响应。
- **灵活路径配置**：支持下拉选择或手动输入自定义路径，适配各种模型存放方式。
- **显存友好**：支持加载到 GPU 或自动 offload 到 CPU，适合不同配置。

## 📦 安装

### 方法一：通过 ComfyUI Manager（推荐）
1. 打开 ComfyUI，点击 Manager → Install Custom Nodes。
2. 搜索 `Qwen3 Translator`，点击 Install。
3. 重启 ComfyUI。

### 方法二：手动安装
1. 克隆本仓库到 `ComfyUI/custom_nodes/` 目录：
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/your-username/ComfyUI_Qwen3_Translator_Node.git
   ```
2. 安装依赖：
   ```bash
   cd ComfyUI_Qwen3_Translator_Node
   pip install -r requirements.txt
   ```
3. 重启 ComfyUI。

## 🧠 模型准备

节点需要 Qwen3 系列模型（如 `Qwen3-8B`、`Qwen3-4B`、`Qwen3.8-27B-FP8` 等），请将模型下载并放置在以下任一目录：
- `ComfyUI/models/text_encoders/`
- `ComfyUI/models/LLM/`

### 下载示例（使用 `hf` 命令）
```bash
# 下载 8B 模型（约 16 GB，适合 24G 显存）
hf download Qwen/Qwen3-8B --local-dir ComfyUI/models/text_encoders/Qwen3-8B

# 下载 4B 模型（约 8 GB，适合低显存）
hf download Qwen/Qwen3-4B --local-dir ComfyUI/models/text_encoders/Qwen3-4B

# 下载 27B FP8 模型（约 56 GB，需大显存或 offload）
hf download Qwen/Qwen3.8-27B-FP8 --local-dir ComfyUI/models/text_encoders/Qwen3.8-27B-FP8
```

> **注意**：如果 `hf` 命令不可用，请先执行 `pip install -U huggingface_hub` 更新。

## 🛠️ 使用方法

1. 在 ComfyUI 节点列表的 **“翻译”** 分类中找到 **“Qwen3 翻译器”**。
2. 添加节点，并根据以下参数配置：

| 参数 | 说明 | 可选值 |
| :--- | :--- | :--- |
| `text` | 待翻译的文本（支持多行） | 任意文本 |
| `source_lang` | 源语言（自动检测或手动选择） | `auto`, `中文`, `English`, `日本語`, `한국어` |
| `target_lang` | 目标语言 | `中文`, `English`, `日本語`, `한국어` |
| `model_choice` | 选择本地扫描到的模型 | 下拉列表自动列出可用模型 |
| `custom_path` | 手动输入模型绝对路径（优先级高于下拉选择） | 留空或填入路径（如 `D:/models/Qwen3-8B`） |
| `max_new_tokens` | 生成的最大 token 数 | 64~4096，默认 512 |
| `temperature` | 生成随机性（0=确定，1=随机） | 0.0~1.0，默认 0.3（适合翻译） |

3. 连接输出 `STRING` 到 `Show Text` 或 `CLIP Text Encode` 等节点查看或使用翻译结果。

## 📝 示例工作流

```
[Load Image] → (可选)
        ↓
[Qwen3 Translator] → (translated_text) → [Show Text] 或 [CLIP Text Encode] → [KSampler]
```

## ⚠️ 注意事项

- **显存需求**：模型大小与显存需求成正比。4B 模型约需 8GB，8B 模型约需 16GB，27B FP8 约需 40GB。24GB 显存建议使用 8B 或更小模型。
- **首次运行**：节点加载模型需要几秒到数十秒，请耐心等待。
- **模型格式**：节点仅支持 Hugging Face 原始格式（包含 `config.json`），不支持 GGUF（如需 GGUF，请使用其他节点）。
- **路径分隔符**：Windows 用户可使用正斜杠 `/` 或双反斜杠 `\\`，避免单反斜杠转义问题。

## 🔧 故障排查

| 问题 | 解决方案 |
| :--- | :--- |
| 下拉列表为空 | 请确保模型已下载并放在 `text_encoders` 或 `LLM` 目录，且包含 `config.json`。 |
| 模型加载失败 | 检查模型文件夹完整性，确认 `config.json` 存在且未损坏。 |
| 显存不足 (OOM) | 更换更小模型（如 4B），或设置 `device_map="auto"`（默认已开启 offload）。 |
| 翻译结果不准确 | 调整 `temperature` 为 0.1~0.3，或更换更大模型。 |

## 📜 许可证

本项目遵循 MIT 许可证，可自由使用、修改和分发。

## 🙏 致谢

- [Qwen (阿里云)](https://huggingface.co/Qwen) 提供强大的开源模型。
- ComfyUI 社区提供的优秀框架。

---

**如果你有任何问题或建议，欢迎在 [GitHub Issues](https://github.com/your-username/ComfyUI_Qwen3_Translator_Node/issues) 提出！**