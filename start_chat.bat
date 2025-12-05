@echo off
echo ========================================
echo Starting Gaia Chat Interface
echo ========================================
echo.
echo Activating virtual environment...
call .venv\Scripts\activate

echo.
echo Starting Streamlit...
cd server
streamlit run scripts/chat.py

pause
