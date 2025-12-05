@echo off
echo ========================================
echo Starting Local Backend with Tunnel
echo ========================================
echo.

echo Step 1: Starting FastAPI Backend...
start "Gaia Backend" cmd /k "cd server && python main.py"

timeout /t 5

echo.
echo Step 2: Starting Cloudflare Tunnel...
echo.
echo This will create a public URL for your backend.
echo Copy the URL and add it to Vercel environment variables.
echo.

cloudflared tunnel --url http://localhost:8000

pause
