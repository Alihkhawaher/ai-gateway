@echo off
cd /d "%~dp0"

echo Starting OpenRouter proxy...
echo   Default model: xiaomi/mimo-v2.5-pro
echo   Listening on:  http://0.0.0.0:8090
echo.
python proxy.py
pause