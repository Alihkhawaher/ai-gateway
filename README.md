# Local AI Gateway — OpenAI-Compatible Proxy with Web Chat UI

A quick-to-launch proxy that serves any OpenAI-compatible AI endpoint to your entire local network. Works with OpenRouter, LM Studio, Ollama, vLLM, LocalAI, and more. No per-client configuration needed — set it up once, and every device on your network can use it.

**Use cases:**
- **Homes** — Share a single paid OpenRouter account with all family devices
- **Testing environments** — Developers and QA can access AI without managing individual API keys
- **Cline / AI agents** — Point any agent client to `http://YOUR_PC:8080` and it just works
- **Private chat** — Anyone on the LAN gets a full chat UI without needing their own account

## What It Does

```
┌─────────────────────────────────────────────────┐
│  Your PC (192.168.1.100:8080)                   │
│                                                  │
│  proxy.py ──► OpenRouter API ──► AI Models       │
│      │                                           │
│      ├── Terminal TUI (model switch, settings)   │
│      └── Web Chat UI (browser-based)             │
│                                                  │
└──────────┬──────────┬──────────┬────────────────┘
           │          │          │
     ┌─────┴──┐  ┌────┴───┐  ┌──┴────────┐
     │ Phone  │  │ Laptop │  │ Cline/Agent│
     │ Web UI │  │ Web UI │  │ API client │
     └────────┘  └────────┘  └────────────┘
```

**One proxy, many clients, zero client configuration.**

### For AI Agent Users (Cline, etc.)
Just change the API base URL to `http://YOUR_PC_IP:8080/v1` — the proxy handles the API key, model selection, and OpenRouter routing. Switch models from the TUI and all connected agents instantly use the new model.

### For Chat Users
Open `http://YOUR_PC_IP:8080/` in any browser on the network. Full chat interface with conversation management, file attachments, markdown rendering, and model switching — no account needed.

## Quick Start

```bash
# 1. Clone or download this project
# 2. Edit config.json — add your OpenRouter API key:
#    "api_key": "sk-or-v1-YOUR_KEY_HERE"
# 3. Run:
python proxy.py

# That's it. Open http://localhost:8080/ for the chat UI.
# Other devices: http://YOUR_IP:8080/
```

## Features

### Proxy Server
- Forwards `/v1/chat/completions` to OpenRouter with SSE streaming
- Automatic model injection when client doesn't specify one
- Real model metadata from OpenRouter (context size, capabilities, modalities)
- CORS headers for cross-origin access
- All models in `config.json` always appear as "loaded" — instant switching

### Terminal Dashboard (TUI)
- Live proxy log showing all requests
- Model selection (applies to all connected agent clients)
- Settings: API key, host, port, bind address
- Save/load settings from `config.json`

### Web Chat UI
- Powered by [llama.cpp's llama-ui](https://github.com/ggml-org/llama.cpp/tree/master/tools/ui)
- Streaming responses with real-time token display
- Model selector with all configured models
- Conversation management (create, branch, edit, delete, search)
- File attachments (images, PDFs, audio, text)
- Markdown with syntax highlighting and math formulas
- Dark/light theme

## Configuration

Edit `config.json`:

```json
{
  "model_index": 0,
  "models": [
    "deepseek/deepseek-v4-flash-0731",
    "openai/gpt-4o",
    "anthropic/claude-sonnet-4",
    "google/gemini-2.5-pro"
  ],
  "api_key": "sk-or-v1-YOUR_OPENROUTER_KEY",
  "port": 8080,
  "addr": "0.0.0.0",
  "host": "openrouter.ai"
}
```

| Field | Description |
|-------|-------------|
| `models` | List of OpenRouter model IDs available in the web UI |
| `model_index` | Default model (0-based index into `models`) |
| `api_key` | Your OpenRouter API key |
| `port` | HTTP port (default: 8080) |
| `addr` | Bind address (`0.0.0.0` = all interfaces) |
| `host` | Upstream API host |

## Connecting Clients

### Cline / AI Agents
Set the API base URL to:
```
http://YOUR_PC_IP:8080/v1
```
No API key needed in the client — the proxy handles it.

### Any OpenAI-compatible client
- **Base URL**: `http://YOUR_PC_IP:8080/v1`
- **API Key**: Leave empty or use anything — the proxy uses its own key

### Web Browser
Open `http://YOUR_PC_IP:8080/`

## Building the Web UI

The web UI only needs to be built once. After building, Python serves the static files directly — no Node.js runtime needed.

```bash
# Option A: Build from cloned llama.cpp source
build-webui.cmd

# Option B: Use pre-built files from a local llama.cpp build
# Copy <llama.cpp>/build/tools/ui/dist/ contents to webui/
```

## Project Structure

```
.
├── proxy.py              # Main application (proxy + TUI + static server)
├── config.json           # Settings (models, API key, port)
├── build-webui.cmd       # One-click web UI build script
├── run.cmd               # Quick-start batch file
├── webui/                # Built web UI static files
├── webui-src/            # llama.cpp source (gitignored)
├── .gitignore
└── README.md
```

## Prerequisites

- **Python 3.8+**
- **Node.js 20+** (only for building the web UI, not for running)

## How Model Metadata Works

On startup, the proxy fetches the full model catalog from OpenRouter and caches it. The web UI shows real context sizes, max tokens, and capabilities for each model.

## LM Studio Compatible

This proxy also works as an LM Studio-compatible API server with a web UI. Any client that expects an OpenAI-compatible endpoint (like LM Studio, Ollama, or LocalAI) can point to this proxy and get access to cloud models through OpenRouter — with the same web chat UI that llama.cpp provides.

**Coming soon:** Model information integration — display model details, capabilities, and parameters directly in the web UI for each configured model.

## License

This project wraps OpenRouter's API. The web UI is from the [llama.cpp](https://github.com/ggml-org/llama.cpp) project (MIT License).
