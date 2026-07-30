#!/bin/bash
# Nightly incremental reindex of the Obsidian vault.
#
# Driven by launchd (com.grahamcarle.obsidian-chat.reindex). Deliberately goes
# through the running server's HTTP API rather than the CLI: ChromaDB is
# SQLite-backed and a second process writing to the same persist directory
# while `serve` and the Claude Desktop MCP server hold it open risks lock
# contention and a stale in-memory collection handle.

set -uo pipefail

HOST="${OBSIDIAN_CHAT_HOST:-127.0.0.1}"
PORT="${OBSIDIAN_CHAT_PORT:-8000}"
BASE_URL="http://${HOST}:${PORT}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Starting nightly reindex against ${BASE_URL}"

# The server is KeepAlive'd but may still be booting the embedding model after
# a reboot, so give it a few tries before giving up.
for attempt in 1 2 3 4 5; do
    if curl -fsS --max-time 10 "${BASE_URL}/health" >/dev/null 2>&1; then
        break
    fi
    if [ "$attempt" -eq 5 ]; then
        log "ERROR: server not reachable at ${BASE_URL} after 5 attempts"
        exit 1
    fi
    log "Server not ready (attempt ${attempt}/5), retrying in 30s"
    sleep 30
done

# No --max-time cap on the reindex itself: a first run that re-embeds the whole
# vault can legitimately take a long time.
response=$(curl -fsS -X POST "${BASE_URL}/index" \
    -H 'Content-Type: application/json' \
    -d '{"force": false}' 2>&1)
status=$?

if [ $status -ne 0 ]; then
    log "ERROR: reindex request failed (curl exit ${status}): ${response}"
    exit 1
fi

log "Reindex complete: ${response}"
