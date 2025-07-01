@echo off
cd /d C:\Users\ihabs\FridayAI

:: Start Ollama server in a separate window if it's not already running
start "Ollama Model" cmd /k "ollama run mistral"

:: Delay to make sure Ollama is ready
timeout /t 5 /nobreak > NUL

:: Run FridayAI in a new window
start "Friday AI" cmd /k "python FridayAI.py"
