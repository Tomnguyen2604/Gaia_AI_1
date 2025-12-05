# 🌍 Gaia Web UI - Complete Setup

Professional web interface for Gaia AI with Next.js + shadcn/ui + FastAPI

## 🚀 Quick Start

### Option 1: Automated (Windows)
```bash
start_web_ui.bat
```

### Option 2: Manual

**Terminal 1 - Backend:**
```bash
cd server
pip install -r requirements.txt
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Then open: **http://localhost:3000**

## 📁 Project Structure

```
├── api/
│   ├── main.py              # FastAPI backend
│   └── requirements.txt     # Python dependencies
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx        # Main chat page
│   │   ├── layout.tsx      # Root layout
│   │   └── globals.css     # Tailwind styles
│   ├── components/
│   │   ├── chat/           # Chat components
│   │   │   ├── message.tsx
│   │   │   ├── chat-input.tsx
│   │   │   └── chat-messages.tsx
│   │   ├── layout/         # Layout components
│   │   │   ├── sidebar.tsx
│   │   │   └── settings-dialog.tsx
│   │   └── ui/             # shadcn/ui components
│   │       ├── button.tsx
│   │       ├── input.tsx
│   │       └── card.tsx
│   ├── lib/
│   │   ├── api.ts          # API client
│   │   └── utils.ts        # Utilities
│   └── package.json
│
└── start_web_ui.bat        # Quick start script
```

## ✨ Features

### Backend (FastAPI)
- ✅ REST API for model inference
- ✅ Auto-loads merged/base model
- ✅ BF16 precision for RTX GPUs
- ✅ CORS enabled
- ✅ Health check endpoints
- ✅ Type-safe with Pydantic

### Frontend (Next.js + shadcn/ui)
- ✅ Modern, professional UI
- ✅ Dark mode with green theme
- ✅ Real-time chat interface
- ✅ Conversation history
- ✅ Settings panel (tokens, temperature, etc.)
- ✅ Responsive design
- ✅ Beautiful animations
- ✅ Type-safe TypeScript

## 🎨 UI Components

### Chat Interface
- Message bubbles with avatars (🌍 Gaia, 👤 User)
- Auto-scrolling to latest message
- Loading states with spinner
- Welcome screen for new chats

### Sidebar
- Conversation list with timestamps
- New chat button
- Delete conversations
- Settings access
- Gaia branding

### Settings Dialog
- Response length slider (512-8192 tokens)
- Temperature control (0.1-1.5)
- Top P sampling (0.1-1.0)
- Repetition penalty (1.0-2.0)

## 🔧 Configuration

### Backend Environment
Create `api/.env`:
```bash
GAIA_MODEL=./gaia-merged
```

### Frontend Environment
Create `frontend/.env.local`:
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 📡 API Endpoints

### GET /
Health check

### GET /health
Detailed status

### POST /chat
Generate response

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "max_tokens": 2048,
  "temperature": 0.7,
  "top_p": 0.9,
  "repetition_penalty": 1.15
}
```

**Response:**
```json
{
  "response": "Hello! I am Gaia...",
  "tokens_generated": 45,
  "generation_time": 1.23
}
```

## 🚀 Production Deployment

### Backend
```bash
# Install production server
pip install gunicorn

# Run with gunicorn
gunicorn api.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend
```bash
cd frontend
npm run build
npm start
```

Or deploy to Vercel:
```bash
vercel deploy
```

## 🐛 Troubleshooting

### API not connecting
- Ensure FastAPI is running: `http://localhost:8000/health`
- Check CORS settings in `api/main.py`
- Verify `.env.local` has correct API URL

### Model not loading
- Check if `gaia-merged` folder exists
- Verify GPU with: `python -c "import torch; print(torch.cuda.is_available())"`
- Ensure 16GB+ VRAM available

### Frontend errors
- Delete `node_modules` and `.next`
- Run `npm install` again
- Check Node.js version (18+ required)

### Port already in use
```bash
# Kill process on port 8000
taskkill /F /IM python.exe

# Kill process on port 3000
taskkill /F /IM node.exe
```

## 📚 Tech Stack

- **Backend**: FastAPI, PyTorch, Transformers
- **Frontend**: Next.js 14, React 18, TypeScript
- **UI**: shadcn/ui, Tailwind CSS, Radix UI
- **Model**: Gemma-2-2B (fine-tuned)

## 🎯 Next Steps

1. ✅ Backend API complete
2. ✅ Frontend UI complete
3. ✅ Chat interface working
4. ✅ Conversation history implemented
5. ✅ Settings panel functional
6. 🔄 Optional: Add authentication
7. 🔄 Optional: Add conversation export
8. 🔄 Optional: Add voice input

## 💡 Tips

- Use `Ctrl+K` to focus chat input (coming soon)
- Dark mode is default (matches Gaia theme)
- Conversations auto-save in memory
- Settings persist during session

Enjoy your professional Gaia web interface! 🌍✨
