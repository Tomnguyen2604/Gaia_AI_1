# 🚀 Deployment Guide

Deploy Gaia with free options: Vercel frontend + Local backend, or Google Colab for demos.

---

## 🎯 Option A: Vercel Frontend + Local Backend (Recommended)

**Best for:** Daily use, maximum performance

### Architecture
```
User → Vercel (Frontend) → Cloudflare Tunnel → Your PC (RTX 5080)
```

### Step 1: Deploy Frontend to Vercel

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy frontend
cd frontend
vercel --prod
```

Or use the batch file:
```bash
deploy_vercel.bat
```

### Step 2: Start Local Backend with Tunnel

```bash
# Start backend and tunnel
start_local_backend.bat
```

This will:
1. Start FastAPI backend on port 8000
2. Create Cloudflare tunnel with public URL
3. Display the tunnel URL in terminal

### Step 3: Configure Vercel Environment

1. Go to [vercel.com/dashboard](https://vercel.com/dashboard)
2. Select your project
3. Go to Settings → Environment Variables
4. Add:
   ```
   NEXT_PUBLIC_API_URL = https://your-tunnel-url.trycloudflare.com
   ```
5. Redeploy (Vercel will auto-redeploy)

### Step 4: Access Your App

- **Frontend:** `https://your-app.vercel.app`
- **Backend:** Running on your PC via tunnel

**Pros:**
- ✅ Best performance (your RTX 5080)
- ✅ Unlimited requests
- ✅ Full control
- ✅ Free

**Cons:**
- ❌ Requires PC to be on
- ❌ Dependent on your internet

---

## 🌐 Option B: Google Colab (For Demos)

**Best for:** Temporary sharing, demos, testing

### Quick Start

1. Open `gaia_colab.ipynb` in [Google Colab](https://colab.research.google.com)
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run all cells
4. Get public URL from last cell
5. Share the URL!

### Manual Setup

```python
# In a new Colab notebook with T4 GPU enabled

# 1. Clone repo
!git clone https://github.com/YOUR_USERNAME/Gaia_AI_1.git
%cd Gaia_AI_1

# 2. Install dependencies
!pip install -q -r requirements.txt

# 3. Login to Hugging Face (for Gemma model)
from huggingface_hub import notebook_login
notebook_login()

# 4. Start Streamlit with public URL
!npm install -g localtunnel
!streamlit run server/scripts/chat.py &>/content/logs.txt & npx localtunnel --port 8501
```

**Pros:**
- ✅ Free T4 GPU
- ✅ Public URL included
- ✅ Easy setup (5 minutes)
- ✅ No PC needed

**Cons:**
- ❌ 12-hour session limit
- ❌ Need to restart manually
- ❌ Slower than your RTX 5080

---

## 📊 Comparison

| Feature | Vercel + Local | Google Colab |
|---------|----------------|--------------|
| **GPU** | RTX 5080 | T4 |
| **Cost** | $0 | $0 |
| **Uptime** | When PC on | 12 hours |
| **Speed** | ⚡⚡⚡ Fastest | ⚡⚡ Good |
| **Setup** | 5 minutes | 5 minutes |
| **Best For** | Daily use | Demos/Sharing |

---

## 🔧 Troubleshooting

### Vercel: CORS Errors

Add to `server/main.py`:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-app.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Cloudflare Tunnel: URL Changes

Free tunnels get new URLs each time. Solutions:
1. Update Vercel env variable with new URL
2. Use persistent tunnel (requires Cloudflare account)

### Colab: Session Expired

Sessions last 12 hours. Just restart the notebook and run all cells again.

---

## 💰 Cost Breakdown

| Component | Cost |
|-----------|------|
| Vercel Frontend | Free |
| Local Backend (your RTX 5080) | Free |
| Cloudflare Tunnel | Free |
| Google Colab | Free (12h sessions) |
| **Total** | **$0** |

**Paid alternatives (if you need 24/7):**
- Google Colab Pro: $10/month (longer sessions)
- Vast.ai GPU: $7-15/month (24/7 hosting)

---

## 🎯 Recommended Workflow

### Daily Use
```bash
# Use Vercel + Local for best performance
start_local_backend.bat
# Access: https://your-app.vercel.app
```

### Demos/Sharing
```
# Use Google Colab when PC is off
# Open gaia_colab.ipynb in Colab
# Share the localtunnel URL
```

---

## 📚 Next Steps

1. ✅ Deploy frontend to Vercel
2. ✅ Set up Cloudflare Tunnel
3. ✅ Test local backend connection
4. ✅ Create Colab notebook for demos
5. ✅ Share your app!

Need help? Check the [README.md](README.md) or [QUICKSTART_DEPLOYMENT.md](QUICKSTART_DEPLOYMENT.md).
