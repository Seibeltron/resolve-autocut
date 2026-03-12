# Resolve Autocut

AI-powered DaVinci Resolve timeline builder. Transcribes a video, lets you pick which topics to include, selects the best segments with GPT-4o storytelling logic, and builds a ready-to-edit timeline in Resolve automatically.

---

## Requirements

- **macOS only** (Resolve's Python scripting bridge is Mac-specific)
- **DaVinci Resolve** installed and running with a project open
- **Resolve scripting enabled**: Preferences → System → General → tick "Enable Resolve scripting via local network"
- **FFmpeg** installed: `brew install ffmpeg`
- **Python 3.12+** installed: `brew install python@3.12`
- **OpenAI API key** set in your shell:
  ```bash
  echo 'export OPENAI_API_KEY="sk-..."' >> ~/.zshrc && source ~/.zshrc
  ```

The `.venv` is created automatically on first run — no other manual setup needed.

---

## How to Use (with Claude Code)

### Step 1 — Clone and open in VS Code

```bash
git clone https://github.com/Seibeltron/resolve-autocut.git ~/resolve-autocut
code ~/resolve-autocut
```

Open the Claude panel (`Cmd+Shift+P` → "Claude: Open") and load the `resolve-autocut` folder.

### Step 2 — Tell Claude to autocut your video

```
Autocut /path/to/my-video.mp4, target 2 minutes
```

Or drag the video file into the chat to get its path.

Claude will:
1. Transcribe the video (1–2 min depending on length)
2. Show a summary and lettered topic categories (A, B, C…)
3. Ask which topics you want — reply e.g. `a c d`
4. Select the best segments using GPT-4o narrative logic
5. Run a trim pass to tighten each clip
6. Build the timeline in Resolve automatically

### Step 3 — Review in Resolve

Switch to the Edit page in DaVinci Resolve. Each segment is a separate clip you can trim, reorder, or swap.

---

## How It Works (Pipeline)

```
transcribe.py   →  segment_select.py  →  trim_pass.py  →  build_timeline.py
  (Whisper)         (GPT-4o picks        (GPT-4o trims     (Resolve Python
  word timestamps   best segments        bridge phrases,   API builds
  via OpenAI API)   w/ 3-act narrative   protects          timeline)
                    structure)           punchlines)
```

### Segment Selection (`segment_select.py`)
GPT-4o reads the full transcript and selects clips using:
- **3-act structure**: Hook → Body → Payoff
- **Transition coherence**: checks every clip boundary for dangling references ("Speaking of X..." requires X to have been mentioned)
- **Shopify context**: merchant-first framing, meaning over metrics
- **Sign-off detection**: closing remarks only appear as the final clip

### Trim Pass (`trim_pass.py`)
After selection, each clip is tightened by GPT-4o:
- Removes bridge phrases from clip starts ("Next up...", "Speaking of...")
- Defers sign-off bleed to the end of the edit
- Protects intentional payoff moments (punchlines, callbacks)
- Stores `trim_note` on each segment so the build step auto-skips the refine pass

---

## Running from the Terminal

```bash
cd ~/resolve-autocut

# Full pipeline manually:

# 1. Transcribe
./run.sh --transcribe /path/to/video.mp4 > /tmp/transcript.json

# 2. Select segments (GPT-4o)
./run.sh --select /tmp/transcript.json \
  --topic "product wins, merchant stories" \
  --duration 120 \
  -o /tmp/segments.json

# 3. Trim pass (GPT-4o)
./run.sh --trim /tmp/segments.json /tmp/transcript.json \
  --keep "any phrase to protect from trimming" \
  -o /tmp/trimmed.json

# 4. Build timeline in Resolve
./run.sh /path/to/video.mp4 /tmp/trimmed.json --timeline-name "My Cut"
```

---

## Troubleshooting

**"Could not connect to DaVinci Resolve"**
→ Make sure Resolve is open with a project loaded. Check that scripting is enabled: Preferences → System → General → "Enable Resolve scripting via local network".

**"OPENAI_API_KEY not set"**
→ Add it to your shell: `echo 'export OPENAI_API_KEY="sk-..."' >> ~/.zshrc && source ~/.zshrc`

**Timeline frame rate is wrong**
→ The script auto-detects fps from the source video. If your Resolve project already has timelines at a different fps, create a new project first.

**Audio missing from clips**
→ This is a known Resolve API quirk — `mediaType: 1` means video-only, not video+audio as documented. The scripts are already set up correctly; if you see this, check you're using `./run.sh` not a direct Python call with custom args.

**Refinement skipped ("faster-whisper not available")**
→ Run via `./run.sh` or `.venv/bin/python3` rather than system Python. The venv has all dependencies installed.
