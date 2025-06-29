@echo off
echo Launching FridayAI...
call .\venv\Scripts\activate
python FridayAI.py
if %errorlevel% neq 0 (
    echo FridayAI exited with error code %errorlevel%.
)
pause
