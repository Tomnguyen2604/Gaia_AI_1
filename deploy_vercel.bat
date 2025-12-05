@echo off
echo ========================================
echo Deploying Frontend to Vercel
echo ========================================
echo.

cd frontend

echo Installing Vercel CLI...
npm install -g vercel

echo.
echo Deploying to Vercel...
echo.
echo IMPORTANT: Set environment variable in Vercel dashboard:
echo   NEXT_PUBLIC_API_URL = your-cloudflare-tunnel-url
echo.

vercel --prod

echo.
echo ========================================
echo Deployment Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Go to vercel.com dashboard
echo 2. Add environment variable: NEXT_PUBLIC_API_URL
echo 3. Set it to your Cloudflare Tunnel URL
echo 4. Redeploy
echo.

pause
