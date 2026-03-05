#!/bin/bash
#
# Start opencode web server + cloudflare tunnel as daemons
#
# Usage:
#   bash start.sh          # Start both services
#   bash start.sh stop     # Stop both services
#   bash start.sh status   # Check status
#
# Access: https://opencode.oneclickamd.ai
#

set -e

OPENCODE_DIR="/home/zijun/opencode-project/opencode/packages/opencode"
CLOUDFLARED="/home/zijun/opencode-project/cloudflared-linux-amd64"
LOG_DIR="/home/zijun/opencode-project/opencode/logs"
OPENCODE_PID_FILE="$LOG_DIR/opencode-web.pid"
TUNNEL_PID_FILE="$LOG_DIR/cloudflared.pid"
PORT=3001

export PATH="$HOME/.bun/bin:$PATH"
export LLM_GATEWAY_KEY="${LLM_GATEWAY_KEY:-1e67b130679141d99d38c6c3b5ddf550}"
export OPENCODE_PERMISSION='{"external_directory":"allow","bash":"allow","edit":"allow","read":"allow","write":"allow","question":"allow","doom_loop":"allow"}'

mkdir -p "$LOG_DIR"

stop_services() {
    echo "Stopping services..."
    if [ -f "$OPENCODE_PID_FILE" ]; then
        kill $(cat "$OPENCODE_PID_FILE") 2>/dev/null && echo "  Stopped opencode web" || echo "  opencode web not running"
        rm -f "$OPENCODE_PID_FILE"
    fi
    if [ -f "$TUNNEL_PID_FILE" ]; then
        kill $(cat "$TUNNEL_PID_FILE") 2>/dev/null && echo "  Stopped cloudflared" || echo "  cloudflared not running"
        rm -f "$TUNNEL_PID_FILE"
    fi
    # Also kill by name as fallback
    pkill -f "opencode.*web.*--port $PORT" 2>/dev/null || true
    pkill -f "cloudflared.*tunnel.*run.*my-tunnel2" 2>/dev/null || true
    echo "Done."
}

check_status() {
    echo "=== Service Status ==="
    if [ -f "$OPENCODE_PID_FILE" ] && kill -0 $(cat "$OPENCODE_PID_FILE") 2>/dev/null; then
        echo "  ✅ opencode web: running (PID $(cat $OPENCODE_PID_FILE))"
    else
        echo "  ❌ opencode web: not running"
    fi
    if [ -f "$TUNNEL_PID_FILE" ] && kill -0 $(cat "$TUNNEL_PID_FILE") 2>/dev/null; then
        echo "  ✅ cloudflared:  running (PID $(cat $TUNNEL_PID_FILE))"
    else
        echo "  ❌ cloudflared:  not running"
    fi
    echo ""
    echo "  URL: https://opencode.oneclickamd.ai"
    echo "  Local: http://localhost:$PORT"
}

start_services() {
    # Stop any existing instances first
    stop_services 2>/dev/null

    echo ""
    echo "╔═══════════════════════════════════════════════╗"
    echo "║  Starting opencode web + cloudflare tunnel    ║"
    echo "╚═══════════════════════════════════════════════╝"
    echo ""

    # 1. Start opencode web server
    echo "Starting opencode web server on port $PORT..."
    cd "$OPENCODE_DIR"
    nohup bun run --conditions=browser ./src/index.ts web \
        --port $PORT \
        --hostname 0.0.0.0 \
        > "$LOG_DIR/opencode-web.log" 2>&1 &
    echo $! > "$OPENCODE_PID_FILE"
    echo "  PID: $(cat $OPENCODE_PID_FILE)"
    echo "  Log: $LOG_DIR/opencode-web.log"

    # Wait for opencode to be ready
    echo "  Waiting for server..."
    for i in $(seq 1 30); do
        if curl -s http://localhost:$PORT > /dev/null 2>&1; then
            echo "  ✅ opencode web ready!"
            break
        fi
        sleep 1
    done

    # 2. Start cloudflare tunnel
    echo ""
    echo "Starting cloudflare tunnel (my-tunnel2 → opencode.oneclickamd.ai)..."
    nohup "$CLOUDFLARED" tunnel run my-tunnel2 \
        > "$LOG_DIR/cloudflared.log" 2>&1 &
    echo $! > "$TUNNEL_PID_FILE"
    echo "  PID: $(cat $TUNNEL_PID_FILE)"
    echo "  Log: $LOG_DIR/cloudflared.log"

    sleep 3

    echo ""
    echo "╔═══════════════════════════════════════════════╗"
    echo "║  ✅ Services started!                         ║"
    echo "║                                               ║"
    echo "║  https://opencode.oneclickamd.ai              ║"
    echo "║  http://localhost:$PORT                        ║"
    echo "║                                               ║"
    echo "║  Logs: $LOG_DIR/                    ║"
    echo "║  Stop: bash start.sh stop                     ║"
    echo "╚═══════════════════════════════════════════════╝"
}

case "${1:-start}" in
    start)   start_services ;;
    stop)    stop_services ;;
    status)  check_status ;;
    restart) stop_services; sleep 2; start_services ;;
    *)       echo "Usage: $0 {start|stop|status|restart}" ;;
esac

