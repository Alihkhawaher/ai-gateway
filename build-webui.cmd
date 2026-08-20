@echo off
setlocal

echo === Building llama.cpp Web UI ===
echo.

set SRC_DIR=%~dp0webui-src\tools\ui
set DIST_DIR=%~dp0webui-src\tools\ui\dist
set DST_DIR=%~dp0webui

:: Guard: webui-src must be a git clone of llama.cpp
if not exist "%~dp0webui-src\.git" (
    echo ERROR: webui-src directory is missing or is not a git clone of llama.cpp.
    echo.
    echo To build the web UI, first clone the llama.cpp source into webui-src:
    echo   git clone https://github.com/ggml-org/llama.cpp.git webui-src
    echo.
    exit /b 1
)
if not exist "%SRC_DIR%" (
    echo ERROR: Could not find llama.cpp web UI source at "%SRC_DIR%".
    echo Make sure webui-src is the llama.cpp repository and the tools/ui folder exists.
    echo.
    exit /b 1
)

:: Step 1: Pull latest changes
echo [1/5] Pulling latest changes from llama.cpp...
cd /d "%~dp0webui-src"
git pull
if errorlevel 1 (
    echo ERROR: git pull failed
    exit /b 1
)

:: Step 2: Install dependencies
echo.
echo [2/5] Installing dependencies...
cd /d "%SRC_DIR%"
call npm install --force
if errorlevel 1 (
    echo ERROR: npm install failed
    exit /b 1
)

:: Step 3: Generate PWA assets (light + dark)
echo.
echo [3/5] Generating PWA assets...
call npx @vite-pwa/assets-generator --root . --config pwa-assets.config.ts
call npx @vite-pwa/assets-generator --root . --config pwa-assets-dark.config.ts

:: Step 4: Build static files
echo.
echo [4/5] Building web UI...
call npx vite build
if errorlevel 1 (
    echo ERROR: vite build failed
    exit /b 1
)

:: Step 5: Copy to project
echo.
echo [5/5] Copying to project...
if exist "%DST_DIR%" rmdir /s /q "%DST_DIR%"
xcopy /s /e /i /q "%DIST_DIR%" "%DST_DIR%"

echo.
echo === Done! Web UI built and copied to %DST_DIR% ===
echo Files:
dir /s /b "%DST_DIR%" | find /c /v ""
echo.