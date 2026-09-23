@echo off
cd /d "%~dp0"

echo Starting AI Gateway (split-view TUI)...
echo   AI Gateway Proxy:  http://localhost:8090  (Web UI: http://localhost:8090/)
echo   1MCP aggregator:   streamable HTTP on :8099/mcp  (aggregates mcp.json)
echo.
python gateway_tui.py
pause
