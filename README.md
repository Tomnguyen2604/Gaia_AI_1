# 🌍 Gaia

**Mother Nature AI** - A nurturing AI assistant powered by Gemma-2-2B.  
Fine-tune via LoRA, chat via Streamlit UI with 32k context window.





## ✅ Features
- ✅ **Gemma-2-2B-IT** with 32k context window
- ✅ LoRA fine-tuning with BF16 precision
- ✅ **Streamlit chat interface** with real-time settings
- ✅ Up to 8192 token responses
- ✅ Multiple dataset support (CSV + HuggingFace)
- ✅ Optimized for RTX GPUs (tested on RTX 5080)
- ✅ Custom identity training (Gaia persona)
- ✅ **Automatic model validation** - detects and backs up corrupted models
- ✅ **Auto-merge after training** - ready to use immediately

## 🚀 Quickstart

### Installation
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA 13.0 (for RTX 5080)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# Login to HuggingFace (for Gemma access)
huggingface-cli login
```

### 💬 Chat with Gaia

**Option 1: Streamlit UI (Recommended)**
```bash
start_chat.bat
# Or: cd server && streamlit run scripts/chat.py
```
Opens at http://localhost:8501
- 🌍 Professional dark mode with green theme
- 💬 Real-time chat with conversation history
- ⚙️ Adjustable settings (tokens, temperature, etc.)
- 📚 Save and load past conversations
- 32K context window, up to 8K token responses

**Option 2: Professional Web UI**
```bash
start_web_ui.bat
```
Opens at http://localhost:3000
- Modern Next.js + shadcn/ui interface
- FastAPI backend
- Production-ready design

### Fine-tuning

**Recommended: Use BF16 for best performance on RTX GPUs**

The training script now includes automatic validation and merging:
1. ✅ Checks if existing merged model is corrupted
2. 📦 Backs up corrupted models automatically
3. 🚀 Trains new LoRA adapter
4. 🔄 Auto-merges with base model
5. ✅ Ready to use immediately!

```bash
# Quick start with batch files (Windows)
train_safe.bat      # BF16 mode (recommended)
train_8bit.bat      # 8-bit mode (lower VRAM)

# Or run directly:
cd server

# With Gaia identity + knowledge datasets (recommended)
python scripts/finetune.py --datasets-file data/datasets_with_identity.txt --bf16

# With single HuggingFace dataset
python scripts/finetune.py --dataset databricks/databricks-dolly-15k --bf16

# With custom CSV only
python scripts/finetune.py --csv data/gaia_identity.csv --bf16

# Skip auto-merge (manual merge later)
python scripts/finetune.py --datasets-file data/datasets_with_identity.txt --bf16 --skip-merge

# Skip validation (trust existing model)
python scripts/finetune.py --datasets-file data/datasets_with_identity.txt --bf16 --skip-validation
```

**Training Configuration:**
- Batch size: 1 (safe for BF16 full precision)
- Gradient accumulation: 16 steps
- Effective batch size: 16
- Precision: BF16 (recommended) or 8-bit
- Expected time: ~30-40 minutes for 4000 samples

**Corrupted Model Detection:**
If your merged model generates gibberish, the training script will:
- Detect the corruption automatically
- Backup the corrupted model with timestamp
- Train and create a fresh merged model

### 🌐 Deploy with Cloudflare Tunnel (Public Access)
```bash
# Terminal 1: Start Streamlit
cd server && streamlit run scripts/chat.py

# Terminal 2: Start Cloudflare Tunnel
.\cloudflared.exe tunnel --url http://localhost:8501
```
Get a public HTTPS URL to share your Gaia instance!