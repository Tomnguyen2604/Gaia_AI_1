# 🌍 Gaia Web UI Setup Guide

Professional web interface with Next.js + shadcn/ui + FastAPI

## Architecture

```
┌─────────────────┐      ┌──────────────┐      ┌─────────────┐
│   Next.js UI    │ ───> │  FastAPI     │ ───> │ Gaia Model  │
│  (Port 3000)    │ HTTP │  (Port 8000) │      │  (GPU)      │
└─────────────────┘      └──────────────┘      └─────────────┘
```

## Quick Start

### 1. Start the Backend Server

```bash
# Install Python dependencies
cd server
pip install -r requirements.txt

# Start FastAPI server
python main.py
```

Server will be available at: `http://localhost:8000`

### 2. Start the Frontend

```bash
# Install Node dependencies
cd frontend
npm install

# Start Next.js dev server
npm run dev
```

Frontend will be available at: `http://localhost:3000`

## Features

### Backend (FastAPI)
- ✅ REST API for model inference
- ✅ CORS enabled for frontend
- ✅ Auto-loads merged model or base model
- ✅ BF16 precision for RTX GPUs
- ✅ Health check endpoints
- ✅ Streaming support (optional)

### Frontend (Next.js + shadcn/ui)
- ✅ Modern, professional UI
- ✅ Dark mode support
- ✅ Real-time chat interface
- ✅ Conversation history
- ✅ Settings panel
- ✅ Responsive design
- ✅ Beautiful animations
- ✅ Markdown support

## API Endpoints

### GET /
Health check

### GET /health
Detailed health status

### POST /chat
Generate chat response

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hello Gaia"}
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

## Environment Variables

### Backend (server/.env)
```bash
GAIA_MODEL=./gaia-merged  # Path to model
```

### Frontend (.env.local)
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Production Deployment

### Backend
```bash
# Using uvicorn
cd server
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

# Using Docker
docker build -t gaia-server -f Dockerfile.server .
docker run -p 8000:8000 --gpus all gaia-server
```

### Frontend
```bash
# Build for production
cd frontend
npm run build
npm start

# Or deploy to Vercel
vercel deploy
```

## Development

### Adding shadcn/ui Components

```bash
cd frontend
npx shadcn-ui@latest add button
npx shadcn-ui@latest add card
npx shadcn-ui@latest add input
# etc...
```

### Project Structure

```
├── server/
│   ├── main.py              # FastAPI application
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── app/
│   │   ├── page.tsx        # Main chat page
│   │   ├── layout.tsx      # Root layout
│   │   └── globals.css     # Global styles
│   ├── components/
│   │   ├── ui/             # shadcn/ui components
│   │   ├── chat/           # Chat components
│   │   └── layout/         # Layout components
│   ├── lib/
│   │   ├── api.ts          # API client
│   │   └── utils.ts        # Utilities
│   └── package.json
└── WEB_UI_SETUP.md         # This file
```

## Troubleshooting

### Server not connecting
- Check if FastAPI is running on port 8000
- Verify CORS settings in `server/main.py`
- Check firewall settings

### Model not loading
- Ensure `gaia-merged` folder exists
- Check GPU availability with `torch.cuda.is_available()`
- Verify VRAM is sufficient (16GB+ recommended)

### Frontend build errors
- Delete `node_modules` and `.next` folders
- Run `npm install` again
- Check Node.js version (18+ required)

## ✅ Complete Setup

### Quick Start (Windows)

```bash
# Run the automated setup
start_web_ui.bat
```

This will:
1. Start FastAPI backend on port 8000
2. Start Next.js frontend on port 3000
3. Open both in separate terminal windows

### Manual Setup

#### Backend Setup
```bash
cd server
pip install -r requirements.txt
python main.py
```

#### Frontend Setup
```bash
cd frontend

# Copy environment file
copy .env.local.example .env.local

# Install dependencies
npm install

# Start development server
npm run dev
```

## 🎨 Features Implemented

### ✅ All 4 Components Complete:

1. **API Integration Layer** (`lib/api.ts`)
   - Type-safe API client
   - Error handling
   - Request/response types

2. **Chat Interface** (`components/chat/`)
   - Message bubbles with avatars
   - Auto-scrolling
   - Loading states
   - Welcome screen

3. **Sidebar** (`components/layout/sidebar.tsx`)
   - Conversation history
   - New chat button
   - Delete conversations
   - Settings access

4. **Main Layout** (`app/page.tsx`)
   - State management
   - Conversation handling
   - Settings dialog
   - Real-time updates

## 🚀 Ready to Use!

Your professional web UI is complete with:
- ✅ Modern shadcn/ui components
- ✅ Dark mode with green theme
- ✅ Responsive design
- ✅ Conversation history
- ✅ Adjustable settings
- ✅ Beautiful animations
- ✅ Type-safe TypeScript

Open http://localhost:3000 and start chatting with Gaia! 🌍
