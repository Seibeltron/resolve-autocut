#!/usr/bin/env bash
# resolve-autocut launcher
# Auto-creates a Python venv with required dependencies on first run.
#
# Full pipeline (single command):
#   ./run.sh <video_or_folder> --topic "Topic" --duration <seconds> [--timeline-name "Name"] [--trim] [--no-cache]
#
# Individual steps:
#   ./run.sh --transcribe <video1> [video2|folder ...] [--no-cache]
#   ./run.sh --select <transcript.json> --topic "Focus topic" --duration <seconds> [-o segments.json]
#   ./run.sh --trim <segments.json> <transcript.json> [--context "note"] [--keep "phrase"] [-o trimmed.json]
#   ./run.sh --build <segments.json> [--timeline-name "Name"] [--no-refine]
set -e
cd "$(dirname "$0")"

PYTHON=python3.12

# Auto-create venv if missing
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv .venv
    .venv/bin/pip install --quiet --upgrade pip
    .venv/bin/pip install --quiet openai
fi

# Install openai if missing (e.g. venv existed from before)
if ! .venv/bin/python3 -c "import openai" 2>/dev/null; then
    echo "Installing openai..."
    .venv/bin/pip install --quiet openai
fi

# Route sub-commands
case "$1" in
    --transcribe)
        shift
        exec .venv/bin/python3 transcribe.py "$@"
        ;;
    --select)
        shift
        exec .venv/bin/python3 segment_select.py "$@"
        ;;
    --trim)
        shift
        exec .venv/bin/python3 trim_pass.py "$@"
        ;;
    --build)
        shift
        exec .venv/bin/python3 build_timeline.py "$@"
        ;;
    --topics)
        # Suggest topics from a video/folder: ./run.sh --topics <video_or_folder> [--n 6] [--no-cache]
        shift
        INPUT="$1"; shift
        N_TOPICS=6
        NO_CACHE=""
        MODEL_ARG=""
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --n) N_TOPICS="$2"; shift 2 ;;
                --no-cache) NO_CACHE="--no-cache"; shift ;;
                --model) MODEL_ARG="--model $2"; shift 2 ;;
                *) echo "Unknown argument: $1" >&2; exit 1 ;;
            esac
        done
        TMPDIR_WORK="$(mktemp -d /tmp/resolve-autocut-XXXXXX)"
        trap 'rm -rf "$TMPDIR_WORK"' EXIT
        TRANSCRIPT="$TMPDIR_WORK/transcript.json"
        .venv/bin/python3 transcribe.py $NO_CACHE "$INPUT" > "$TRANSCRIPT"
        .venv/bin/python3 segment_select.py "$TRANSCRIPT" --suggest-topics --n-topics "$N_TOPICS" $MODEL_ARG
        ;;
    *)
        # Full pipeline: <video_or_folder> --topic "..." --duration <s> [--timeline-name "..."] [--trim] [--no-cache] [--no-refine]
        INPUT="$1"
        shift

        # Parse pipeline-specific flags, pass the rest through to sub-commands
        TOPIC=""
        DURATION=""
        TIMELINE_NAME=""
        DO_TRIM=0
        NO_CACHE=""
        NO_REFINE=""
        EXTRA_SELECT_ARGS=""

        while [[ $# -gt 0 ]]; do
            case "$1" in
                --topic)       TOPIC="$2";         shift 2 ;;
                --duration)    DURATION="$2";      shift 2 ;;
                --timeline-name) TIMELINE_NAME="$2"; shift 2 ;;
                --trim)        DO_TRIM=1;           shift ;;
                --no-cache)    NO_CACHE="--no-cache"; shift ;;
                --no-refine)   NO_REFINE="--no-refine"; shift ;;
                --no-cold-open) EXTRA_SELECT_ARGS="$EXTRA_SELECT_ARGS --no-cold-open"; shift ;;
                --mix)         EXTRA_SELECT_ARGS="$EXTRA_SELECT_ARGS --mix"; shift ;;
                --model)       EXTRA_SELECT_ARGS="$EXTRA_SELECT_ARGS --model $2"; shift 2 ;;
                --no-prescore) EXTRA_SELECT_ARGS="$EXTRA_SELECT_ARGS --no-prescore"; shift ;;
                --prescore-threshold) EXTRA_SELECT_ARGS="$EXTRA_SELECT_ARGS --prescore-threshold $2"; shift 2 ;;
                --prescore-model) EXTRA_SELECT_ARGS="$EXTRA_SELECT_ARGS --prescore-model $2"; shift 2 ;;
                --interactive|-i) EXTRA_SELECT_ARGS="$EXTRA_SELECT_ARGS --interactive"; shift ;;
                *) echo "Unknown argument: $1" >&2; exit 1 ;;
            esac
        done

        if [ -z "$DURATION" ]; then
            echo "Usage: ./run.sh <video_or_folder> --duration <seconds> [--topic \"Topic\"]" >&2
            echo "       [--timeline-name \"Name\"] [--trim] [--no-cache] [--no-refine]" >&2
            exit 1
        fi

        TMPDIR_WORK="$(mktemp -d /tmp/resolve-autocut-XXXXXX)"
        trap 'rm -rf "$TMPDIR_WORK"' EXIT

        TRANSCRIPT="$TMPDIR_WORK/transcript.json"
        SEGMENTS="$TMPDIR_WORK/segments.json"
        TRIMMED="$TMPDIR_WORK/trimmed.json"

        echo "=== Step 1/3: Transcribe ===" >&2
        .venv/bin/python3 transcribe.py $NO_CACHE "$INPUT" > "$TRANSCRIPT"

        echo "=== Topics ===" >&2
        .venv/bin/python3 segment_select.py "$TRANSCRIPT" --suggest-topics

        if [ -z "$TOPIC" ]; then
            echo "" >&2
            echo "No --topic provided. Choose a topic above and re-run with --topic \"...\"" >&2
            exit 0
        fi

        echo "=== Step 2/3: Select segments ===" >&2
        .venv/bin/python3 segment_select.py "$TRANSCRIPT" \
            --topic "$TOPIC" --duration "$DURATION" \
            $EXTRA_SELECT_ARGS \
            -o "$SEGMENTS"

        if [ "$DO_TRIM" -eq 1 ]; then
            echo "=== Step 2b: Trim pass ===" >&2
            .venv/bin/python3 trim_pass.py "$SEGMENTS" "$TRANSCRIPT" -o "$TRIMMED"
            SEGMENTS="$TRIMMED"
        fi

        echo "=== Step 3/3: Build Resolve timeline ===" >&2
        TL_ARG=""
        [ -n "$TIMELINE_NAME" ] && TL_ARG="--timeline-name \"$TIMELINE_NAME\""
        .venv/bin/python3 build_timeline.py $NO_REFINE "$SEGMENTS" ${TIMELINE_NAME:+--timeline-name "$TIMELINE_NAME"}
        ;;
esac
