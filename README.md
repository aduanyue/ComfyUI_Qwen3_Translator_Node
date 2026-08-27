# Qwen3 Translator Node for ComfyUI

一个基于 **Qwen3** 系列模型的 ComfyUI 本地翻译节点，支持**纯文本模型**和**视觉语言模型（VL）**，自动识别模型类型并智能提取翻译结果。

## ✨ 特性

- **自动模型类型识别**  
  自动检测模型是纯文本（如 `Qwen3-4B`）还是视觉语言模型（如 `Qwen3-VL-4B`），并使用对应的加载方式（`AutoModelForCausalLM` 或 `Qwen3VLForConditionalGeneration`）。

- **多语言翻译**  
  支持中文、英文、日文、韩文互译，源语言可设置为自动检测。

- **智能结果提取**  
  针对不同模型输出格式，自动提取翻译内容，去除指令、标记和多余描述，保留 Markdown 格式（如 `**粗体**`、`---`、列表等）。

- **完全本地运行**  
  无需联网，保护隐私，所有推理在本地完成。

- **自动模型扫描**  
  自动扫描 `text_encoders`、`LLM`、`prompt_generator` 目录下的模型文件夹（只要包含 `config.json`），下拉菜单直接选择。

- **模型兼容性检查**  
  对 `Qwen3.5-9B` 等需要新版 `transformers` 的模型会给出清晰的升级提示。

## 📦 安装

### 方法一：通过 ComfyUI Manager（推荐）

1. 在 ComfyUI 界面中点击 **Manager** 按钮。
2. 选择 **“Install Custom Nodes”**。
3. 搜索 **`Qwen3 Translator`**，点击 **Install**。
4. 重启 ComfyUI。

### 方法二：手动安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/你的用户名/ComfyUI_Qwen3_Translator_Node.git
cd ComfyUI_Qwen3_Translator_Node
pip install -r requirements.txt