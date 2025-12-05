@echo off
echo ========================================
echo Starting Gaia Web UI
echo ========================================
echo.

echo Step 1: Starting FastAPI Backend...
start "Gaia Server" cmd /k "cd server && .venv\Scripts\activate && python main.py"
timeout /t 5

echo.
echo Step 2: Starting Next.js Frontend...
start "Gaia UI" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================
echo Gaia Web UI is starting!
echo ========================================
echo.
echo Server: http://localhost:8000
echo UI:     http://localhost:3000
echo.
echo Press any key to stop all services...
pause > nul

taskkill /FI "WINDOWTITLE eq Gaia Server*" /F
taskkill /FI "WINDOWTITLE eq Gaia UI*" /F
