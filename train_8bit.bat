@echo off
echo ========================================
echo Gaia Training - 8-bit Mode
echo ========================================
echo.
echo This will:
echo 1. Check if existing merged model is corrupted
echo 2. Backup corrupted model if found
echo 3. Train new LoRA adapter (8-bit, lower VRAM)
echo 4. Auto-merge with base model
echo.
call .venv\Scripts\activate
cd server
python scripts/finetune.py --datasets-file data/datasets_with_identity.txt --load_in_8bit
pause
