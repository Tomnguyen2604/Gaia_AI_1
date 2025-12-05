# ⚡ Quick Start: Deployment

Get Gaia deployed in 10 minutes!

---

## 🚀 5-Minute Setup: Vercel + Local

### 1. Deploy Frontend (2 minutes)

```bash
cd frontend
npm install -g vercel
vercel --prod
```

**Result:** Your frontend is live at `https://your-app.vercel.app`

### 2. Start Local Backend (1 minute)

```bash
# Run this batch file
start_local_backend.bat
```

**Result:** Backend running with public tunnel URL

### 3. Connect Them (2 minutes)

1. Copy your Cloudflare tunnel URL (from terminal)
2. Go to [vercel.com/dashboard](https://vercel.com/dashboard)
3. Your Project → Settings → Environment Variables
4. Add: `NEXT_PUBLIC_API_URL` = `your-tunnel-url`
5. Click "Redeploy"

**Done!** Your app is live! 🎉

---

## 🌐 Alternative: Google Colab (For Demos)

For temporary sharing when your PC is off:

### Quick Start (5 minutes)

1. Open `gaia_colab.ipynb` in [Google Colab](https://colab.research.google.com)
2. Runtime → Change runtime type → T4 GPU
3. Run all cells
4. Get public URL from last cell
5. Share the URL!

**Note:** Session lasts 12 hours, then you need to restart.

---

## 📱 Usage

### Daily Use (Best Performance)
```
https://your-app.vercel.app
↓
Your RTX 5080 (via tunnel)
```

### Sharing/Demo (Temporary)
```
Google Colab Notebook
↓
Free T4 GPU (12-hour sessions)
```

---

## 🔧 Common Issues

### "API Connection Failed"
- Check if backend is running: `http://localhost:8000/health`
- Check if tunnel is active
- Update Vercel environment variable

### "Tunnel URL Changed"
Cloudflare free tunnels change URLs. Solutions:
1. Update Vercel env variable with new URL
2. Use persistent tunnel (requires Cloudflare account)

### "Colab Session Expired"
Free Colab sessions last 12 hours. Just restart the notebook and run cells again.

---

## 💡 Pro Tips

1. **Keep PC on for best performance**
   - Use Vercel + Local for daily use
   - Your RTX 5080 is much faster

2. **Use Google Colab for sharing**
   - Free T4 GPU for 12 hours
   - Easy to restart
   - No PC needed

3. **Update deployment**
   ```bash
   # Update Vercel (auto-deploys)
   git push origin main
   ```

---

## 🎉 You're Done!

You now have:
- ✅ Professional frontend on Vercel
- ✅ Powerful backend on your RTX 5080
- ✅ Demo option via Google Colab
- ✅ Total cost: $0

Enjoy your deployed Gaia! 🌍✨
