# Obsidian Chat

CLI tool and web server for chatting with a local LLM using RAG (Retrieval-Augmented Generation) from your Obsidian vault.

## Features

- **RAG-powered chat** - Ask questions and get answers informed by your Obsidian notes
- **Local LLM support** - Works with any OpenAI-compatible API (LM Studio, Ollama, etc.)
- **Web UI** - Browser-based chat interface at http://127.0.0.1:8000
- **CLI tools** - Index, query, and chat from the command line
- **Auto-start** - Can be configured to run as a background service on macOS

## Installation

```bash
pip install -e .
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# LLM API settings (OpenAI-compatible endpoint)
LLM_BASE_URL=http://localhost:1234/v1
LLM_MODEL=local-model
LLM_API_KEY=not-needed

# Path to your Obsidian vault
OBSIDIAN_VAULT_PATH=/path/to/your/obsidian/vault

# ChromaDB settings (optional)
CHROMA_PERSIST_DIR=~/.obsidian-chat/chroma
CHROMA_COLLECTION=obsidian_notes

# Embedding model (optional - uses all-MiniLM-L6-v2 by default)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# RAG settings
RAG_TOP_K=5
```

## CLI Usage

```bash
# Index your vault (required before first use)
obsidian-chat index

# Re-index with force flag
obsidian-chat index --force

# Show index statistics
obsidian-chat stats

# Interactive chat session
obsidian-chat chat

# Ask a single question
obsidian-chat ask "What notes do I have about Python?"

# Search notes without LLM (RAG only)
obsidian-chat query "Python"
```

## Web Server

Start the web server with API and browser-based UI:

```bash
# Start on default port 8000
obsidian-chat serve

# Custom host and port
obsidian-chat serve --host 127.0.0.1 --port 8080
```

Then open http://127.0.0.1:8000 in your browser.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Health check and configuration status |
| `/stats` | GET | Index statistics |
| `/index` | POST | Index/reindex the vault |
| `/query` | POST | Query notes (RAG only, no LLM) |
| `/chat` | POST | Chat with LLM + RAG context |
| `/chat/stream` | POST | Streaming chat response |

## Auto-Start on macOS

To run the server automatically when you log in, create a LaunchAgent:

### 1. Create the plist file

Create `~/Library/LaunchAgents/com.obsidian-chat.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.obsidian-chat</string>

    <key>ProgramArguments</key>
    <array>
        <string>/path/to/your/venv/bin/obsidian-chat</string>
        <string>serve</string>
        <string>--host</string>
        <string>127.0.0.1</string>
        <string>--port</string>
        <string>8000</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/path/to/obsidian-chat</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>/Users/YOU/Library/Logs/obsidian-chat.log</string>

    <key>StandardErrorPath</key>
    <string>/Users/YOU/Library/Logs/obsidian-chat.error.log</string>
</dict>
</plist>
```

Update the paths to match your installation.

### 2. Load the service

```bash
# Load and start
launchctl load ~/Library/LaunchAgents/com.obsidian-chat.plist

# Check status
launchctl list | grep obsidian-chat

# Stop the service
launchctl unload ~/Library/LaunchAgents/com.obsidian-chat.plist
```

### 3. View logs

```bash
tail -f ~/Library/Logs/obsidian-chat.log
tail -f ~/Library/Logs/obsidian-chat.error.log
```

## Architecture

- **FastAPI** - Web server and API
- **ChromaDB** - Vector database for embeddings
- **sentence-transformers** - Text embeddings (all-MiniLM-L6-v2)
- **Typer** - CLI framework
