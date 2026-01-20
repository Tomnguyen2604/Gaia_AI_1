# 🌍 Gaia

**Mother Nature AI** - A nurturing AI assistant powered by Gemma-2-2B.  
Fine-tune via LoRA, chat via Streamlit UI with 32k context window.

## 📁 Project Structure

```
Gaia/
├── server/              # AI Backend (all AI components)
│   ├── scripts/        # Python scripts (training, chat, utilities)
│   ├── data/           # Training datasets
│   ├── gaia-merged/    # Fine-tuned models (excluded from git)
│   ├── templates/      # Jinja2 templates
│   ├── main.py         # FastAPI backend
│   └── rag_system.py   # RAG implementation
├── frontend/           # Next.js Web UI
├── .venv/             # Python virtual environment
├── *.bat              # Quick start scripts
└── *.md               # Documentation
```

## ✅ Features

### Core AI
- ✅ **Gemma-2-2B-IT** with 32k context window
- ✅ LoRA fine-tuning with BF16 precision
- ✅ Up to 8192 token responses
- ✅ Optimized for RTX GPUs (tested on RTX 5080)
- ✅ Custom identity training (Gaia persona)
- ✅ **Automatic model validation** - detects and backs up corrupted models
- ✅ **Auto-merge after training** - ready to use immediately

### User Interfaces
- ✅ **Streamlit chat interface** with professional dark mode
- ✅ **Next.js Web UI** with shadcn/ui components
- ✅ Real-time settings and conversation history
- ✅ Voice input support
- ✅ Conversation export (Markdown, JSON, Text)

### Advanced Features
- ✅ **RAG System** - Retrieval-Augmented Generation with scientific citations
- ✅ **GraphQL API** - Real-time streaming with subscriptions and flexible queries
- ✅ Multiple dataset support (CSV + HuggingFace)
- ✅ FastAPI backend for production deployment
- ✅ Git LFS support for large model files

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

### 🧠 RAG System (Optional)

Add knowledge retrieval with scientific citations:

```bash
cd server

# Install RAG dependencies
pip install sentence-transformers chromadb

# Build knowledge base (one-time setup)
python scripts/build_knowledge_base.py

# Start server with RAG enabled
python main.py
```

See [RAG_SETUP.md](RAG_SETUP.md) for detailed instructions.

### 🌐 Deploy with Cloudflare Tunnel (Public Access)
```bash
# Terminal 1: Start Streamlit
cd server && streamlit run scripts/chat.py

# Terminal 2: Start Cloudflare Tunnel
.\cloudflared.exe tunnel --url http://localhost:8501
```
Get a public HTTPS URL to share your Gaia instance!

## 🚀 Deployment

Deploy Gaia to the cloud for 24/7 access:

- **[QUICKSTART_DEPLOYMENT.md](QUICKSTART_DEPLOYMENT.md)** - 10-minute deployment guide ⚡
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Complete deployment documentation

**Quick Deploy:**
```bash
# Option 1: Vercel Frontend + Local Backend (Best Performance)
deploy_vercel.bat
start_local_backend.bat

# Option 2: Hugging Face Spaces (24/7 Availability)
# See QUICKSTART_DEPLOYMENT.md
```

## 📚 Documentation

- **[GRAPHQL_SETUP.md](GRAPHQL_SETUP.md)** - GraphQL API setup and usage guide
- **[RAG_SETUP.md](RAG_SETUP.md)** - RAG system setup and configuration
- **[WEB_UI_SETUP.md](WEB_UI_SETUP.md)** - Web UI installation guide
- **[WEB_UI_README.md](WEB_UI_README.md)** - Web UI features and usage
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Cloud deployment guide

## 🛠️ Development

### Project Organization

All AI-related code is in the `server/` directory:
- `server/scripts/` - Training, chat, and utility scripts
- `server/data/` - Training datasets and configuration
- `server/gaia-merged/` - Fine-tuned models (not in git)
- `server/main.py` - FastAPI backend server
- `server/rag_system.py` - RAG implementation

### Running Scripts

Always run scripts from the `server/` directory:

```bash
cd server

# Training
python scripts/finetune.py --datasets-file data/datasets_with_identity.txt --bf16

# Chat
streamlit run scripts/chat.py

# RAG setup
python scripts/build_knowledge_base.py

# FastAPI server
python main.py
```

Or use the convenient batch files from the root:
```bash
start_chat.bat      # Streamlit UI
start_web_ui.bat    # Next.js + FastAPI
train_safe.bat      # Training (BF16)
train_8bit.bat      # Training (8-bit)
```

## 🤝 Contributing

This project uses:
- **Python 3.11+** for AI backend
- **Next.js 14** for web frontend
- **Git LFS** for large model files

Model files are excluded from git via `.gitignore`. Only code and configuration are tracked.

## 📄 License

This project uses the Gemma-2-2B model which requires acceptance of Google's terms.