# 🌍 Gaia-RTX

**NVIDIA-optimized LLM** for RTX GPUs with extended context support.  
Fine-tune via LoRA, chat via Streamlit UI with 32k context window.



## ✅ Features
- ✅ **Gemma-2-2B-IT** with 32k context window
- ✅ LoRA fine-tuning (4-bit quantization)
- ✅ **Streamlit chat interface** with auto-reload
- ✅ Up to 8192 token responses
- ✅ CPU offloading support (128GB RAM)
- ✅ Multiple dataset support via text file
- ✅ RTX 5080 optimized

## 🚀 Quickstart

### Installation
```bash
pip install -r requirements.txt
pip install streamlit

# Login to HuggingFace (for Gemma access)
huggingface-cli login
```

### Chat Interface (Recommended)
```bash
streamlit run Scripts/chat_streamlit.py
```
Opens at http://localhost:8501 with:
- 32k token context window
- Adjustable response length (up to 8192 tokens)
- Real-time settings control
- Auto-reload on code changes

### Fine-tuning
```bash
# With single dataset
python Scripts/finetune.py --dataset databricks/databricks-dolly-15k

# With multiple datasets from file
python Scripts/finetune.py --datasets-file data/datasets.txt

# With custom CSV
python Scripts/finetune.py --csv data/clean_chat_data.csv
```

### Alternative: Gradio Chat
```bash
python Scripts/chat.py
```
Opens at http://localhost:7860