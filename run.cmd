@echo off
cd /d "%~dp0"

echo Starting AI Gateway (split-view TUI)...
echo   AI Gateway Proxy:  http://localhost:8090  (Web UI: http://localhost:8090/)
echo   Supergateway MCP:  streamable HTTP on :8099  (filesystem root: .\supergateway)
echo.
python gateway_tui.py
pause
