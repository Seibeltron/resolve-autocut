#!/usr/bin/env bash
# setup.sh — One-command setup for resolve-autocut + DaVinci Resolve MCP
# Usage: bash setup.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC}  $1"; }
err()  { echo -e "${RED}✗${NC} $1"; exit 1; }
step() { echo -e "\n${BOLD}$1${NC}"; }

echo -e "${BOLD}Resolve Autocut — Setup${NC}"
echo "────────────────────────────────────────"

# ── 1. FFmpeg ─────────────────────────────────────────────────────────────────
step "1/5  Checking FFmpeg..."
if command -v ffmpeg &>/dev/null; then
    ok "FFmpeg already installed"
else
    if ! command -v brew &>/dev/null; then
        err "Homebrew not found. Install it first: https://brew.sh"
    fi
    echo "    Installing FFmpeg via Homebrew..."
    brew install ffmpeg
    ok "FFmpeg installed"
fi

# ── 2. OpenAI API key ─────────────────────────────────────────────────────────
step "2/5  OpenAI API key..."
if [[ -n "$OPENAI_API_KEY" ]]; then
    ok "OPENAI_API_KEY already set"
else
    echo ""
    echo "    Get your key from: https://openai-proxy.shopify.io/dashboard"
    echo "    Click 'Generate Key', then paste it below."
    echo ""
    read -r -p "    Paste your API key: " api_key
    if [[ -z "$api_key" ]]; then
        err "No API key provided. Re-run setup.sh after getting your key."
    fi
    # Write to .zshrc
    echo "" >> ~/.zshrc
    echo "# Shopify OpenAI proxy (added by resolve-autocut setup)" >> ~/.zshrc
    echo "export OPENAI_API_KEY=\"$api_key\"" >> ~/.zshrc
    # Also export for current session
    export OPENAI_API_KEY="$api_key"
    ok "API key saved to ~/.zshrc"
fi

# ── 3. Python venv for resolve-autocut ───────────────────────────────────────
step "3/5  Setting up Python environment..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
fi
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt
ok "Python environment ready"

# ── 4. DaVinci Resolve MCP ───────────────────────────────────────────────────
step "4/5  Installing DaVinci Resolve MCP server..."
MCP_DIR="$HOME/davinci-resolve-mcp"

if [[ -d "$MCP_DIR" ]]; then
    ok "davinci-resolve-mcp already cloned"
else
    echo "    Cloning davinci-resolve-mcp..."
    git clone https://github.com/samuelgursky/davinci-resolve-mcp.git "$MCP_DIR"
    ok "Cloned to $MCP_DIR"
fi

echo "    Running MCP installer for Claude Code..."
cd "$MCP_DIR"

# Create venv for MCP server if needed
if [[ ! -d "venv" ]]; then
    python3 -m venv venv
    venv/bin/pip install --quiet --upgrade pip
    venv/bin/pip install --quiet mcp
fi

# Run the installer non-interactively for claude-code
python3 install.py --clients claude-code 2>&1 | grep -E "✓|✗|Configured|Skipped|already|Error" || true
ok "DaVinci Resolve MCP configured for Claude Code"

# ── 5. Resolve scripting reminder ────────────────────────────────────────────
step "5/5  DaVinci Resolve scripting..."
echo ""
warn "One manual step required in DaVinci Resolve:"
echo ""
echo "    DaVinci Resolve → Preferences → System → General"
echo "    ✅ Tick: 'Enable Resolve scripting via local network'"
echo "    → Save & restart Resolve"
echo ""

# ── Done ──────────────────────────────────────────────────────────────────────
echo "────────────────────────────────────────"
echo -e "${GREEN}${BOLD}Setup complete!${NC}"
echo ""
echo "  Reload your shell:  source ~/.zshrc"
echo ""
echo "  Then open VS Code:"
echo "    code $SCRIPT_DIR"
echo ""
echo "  Tell Claude:"
echo "    Autocut /path/to/your-video.mp4, target 2 minutes"
echo ""
