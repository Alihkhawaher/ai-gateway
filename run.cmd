@echo off
cd /d "%~dp0"

echo Starting AI Gateway...
echo   Multi-endpoint proxy with health checking
echo   Listening on:  http://0.0.0.0:8090
echo   Web UI:        http://localhost:8090/
echo.
python proxy.py
pause