# 🧠 RAG System Setup for Gaia

Retrieval-Augmented Generation adds knowledge retrieval to enhance Gaia's responses with scientific citations.

## 🎯 What is RAG?

RAG combines:
1. **Retrieval** - Search knowledge base for relevant info
2. **Augmentation** - Add retrieved context to user query
3. **Generation** - Your Gemma-2-2B generates response with context

**Result:** More accurate, cited responses with scientific backing!

## 📦 Installation

```bash
cd server
pip install sentence-transformers chromadb
```

## 🏗️ Build Knowledge Base

**One-time setup:**
```bash
cd server
python scripts/build_knowledge_base.py
```

This will:
- Load natural medicine articles (400 examples) from `data/`
- Load comprehensive dataset (5,000 examples) from `data/`
- Load Gaia identity data from `data/`
- Create vector embeddings
- Store in `server/knowledge_base/` folder

**Output:**
```
✅ Knowledge base built successfully!
📊 Total documents: 5,400+
💾 Stored in: knowledge_base/
```

## 🚀 Usage

### Start Server with RAG
```bash
cd server
python main.py
```

The server will:
1. Load your Gaia model (~5-6GB VRAM)
2. Load embedding model (~500MB VRAM)
3. Connect to vector database
4. **Total VRAM: ~6.5GB** (plenty of room!)

### Check RAG Status
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "rag_enabled": true,
  "rag_documents": 5400,
  "device": "cuda:0"
}
```

## 💬 How It Works

### Without RAG:
```
User: "What are the benefits of turmeric?"
  ↓
Gaia: [Generates from training data only]
```

### With RAG:
```
User: "What are the benefits of turmeric?"
  ↓
RAG: [Searches knowledge base]
  ↓
Found: "Turmeric (Curcuma longa) has anti-inflammatory..."
Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5664031/
  ↓
Gaia: [Generates with retrieved context + citation]
```

## 🎛️ Configuration

### Enable/Disable RAG per request

**Frontend (automatic):**
RAG is enabled by default for all queries.

**API (manual):**
```json
{
  "messages": [...],
  "use_rag": true  // Set to false to disable
}
```

### Adjust retrieval count

In `server/main.py`, change:
```python
rag_context = rag_system.build_context(user_query, n_results=2)
```

- `n_results=1` - Single best match (faster)
- `n_results=2` - Two sources (balanced) ⭐ Default
- `n_results=3` - Three sources (more context)

## 📊 Performance Impact

### VRAM Usage:
- **Before RAG:** ~5-6GB
- **After RAG:** ~6.5GB
- **Increase:** ~500MB (minimal!)

### Response Time:
- **Retrieval:** ~50-100ms
- **Generation:** ~1-3 seconds (same as before)
- **Total increase:** ~50-100ms (barely noticeable!)

### Quality Improvement:
- ✅ More accurate medical information
- ✅ Scientific citations included
- ✅ Up-to-date knowledge from database
- ✅ Reduced hallucinations

## 🔧 Troubleshooting

### "RAG system failed to load"
- Check if `sentence-transformers` is installed
- Verify `chromadb` is installed
- Ensure knowledge base files exist

### "No documents in knowledge base"
- Run `python build_knowledge_base.py`
- Check if CSV files exist in `data/` folder
- Verify file paths in `build_knowledge_base.py`

### "Out of memory"
- RAG uses ~500MB extra VRAM
- Should fit easily on RTX 5080 (16GB)
- If issues, reduce `n_results` to 1

## 📚 Adding More Knowledge

### Add new CSV files:
```bash
# In server/data/datasets.txt, add:
your_new_file.csv
another_file.csv
```

The build script automatically loads all CSV files listed in `data/datasets.txt`

### Add individual documents:
```python
from rag_system import GaiaRAG

rag = GaiaRAG()
rag.add_document(
    text="Your knowledge here...",
    metadata={"source": "custom", "topic": "herbs"},
    doc_id="custom_1"
)
```

### Rebuild knowledge base:
```bash
cd server

# Delete old database
rm -rf knowledge_base/

# Rebuild
python scripts/build_knowledge_base.py
```

## 🎯 Benefits

### For Users:
- ✅ More accurate health information
- ✅ Scientific citations and sources
- ✅ Reduced misinformation
- ✅ Trustworthy responses

### For You:
- ✅ Easy to update knowledge (just add CSV files)
- ✅ No retraining needed
- ✅ Minimal VRAM overhead
- ✅ Fast retrieval (<100ms)

## 🚀 Next Steps

1. **Build knowledge base:**
   ```bash
   cd server
   python scripts/build_knowledge_base.py
   ```

2. **Start server:**
   ```bash
   python main.py
   ```

3. **Test RAG:**
   Ask Gaia: "What are the benefits of turmeric?"
   
   Response should include:
   - Detailed information
   - Scientific backing
   - Source citation

4. **Add more knowledge:**
   - Create new CSV files in `data/`
   - Run `build_knowledge_base.py` again
   - Restart server

## 💡 Tips

- **Keep knowledge focused:** Natural medicine, wellness, environment
- **Update regularly:** Rebuild knowledge base when adding new data
- **Monitor quality:** Check if citations are relevant
- **Adjust n_results:** More sources = more context but slower

Your Gaia now has a knowledge base! 🌍✨
