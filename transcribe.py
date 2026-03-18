#!/usr/bin/env python3
"""
transcribe.py — Word-accurate transcription using OpenAI Whisper API.
Outputs JSON to stdout: {segments, words, total_duration, method, sources}

Each segment and word is tagged with source_video (absolute path) to support
multi-clip workflows where segments from different files are merged.

Results are cached in ~/.cache/resolve-autocut/ keyed by file path + mtime.
Re-running on the same file returns the cached result instantly.

Usage:
    python transcribe.py /path/to/video.mp4
    python transcribe.py /path/to/video.mp4 > transcript.json
    python transcribe.py /path/to/video.mp4 --no-cache
    python transcribe.py video1.mp4 video2.mp4 > combined.json
    python transcribe.py /path/to/folder/ > combined.json
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List


WORD_PAD_S = 0.05  # 50ms padding on each word boundary
CACHE_DIR = Path.home() / ".cache" / "resolve-autocut"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".m4v", ".mts", ".avi", ".webm"}


def expand_paths(paths: List[str]) -> List[str]:
    """Expand any directory paths to sorted lists of video files within them."""
    expanded = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files = sorted(
                f for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
            )
            if not files:
                print(f"Warning: no video files found in {path}", file=sys.stderr)
            expanded.extend(str(f) for f in files)
        else:
            expanded.append(p)
    return expanded


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _cache_key(video_path: str) -> str:
    """Return a stable cache key based on absolute path + mtime + size."""
    p = Path(video_path).resolve()
    stat = p.stat()
    raw = f"{p}:{stat.st_mtime}:{stat.st_size}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_path(video_path: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(video_path)
    name = Path(video_path).stem[:40]
    return CACHE_DIR / f"{name}_{key}.json"


def load_cached(video_path: str) -> Dict | None:
    cp = _cache_path(video_path)
    if cp.exists():
        try:
            return json.loads(cp.read_text())
        except Exception:
            cp.unlink(missing_ok=True)
    return None


def save_cache(video_path: str, result: Dict) -> None:
    try:
        cp = _cache_path(video_path)
        cp.write_text(json.dumps(result))
    except Exception:
        pass  # cache write failure is non-fatal


# ---------------------------------------------------------------------------
# Segment building / scoring
# ---------------------------------------------------------------------------

def _pad_words(words: List[Dict]) -> List[Dict]:
    return [
        {**w, "start": max(0.0, w["start"] - WORD_PAD_S), "end": w["end"] + WORD_PAD_S}
        for w in words
    ]


def _words_to_segments(words: List[Dict], pause_gap: float = 0.7) -> List[Dict]:
    if not words:
        return []
    segments = []
    cur = [words[0]]
    for w in words[1:]:
        gap = w["start"] - cur[-1]["end"]
        ends_sentence = cur[-1].get("word", "").strip().endswith((".", "!", "?"))
        if gap >= pause_gap or ends_sentence:
            segments.append(_finalize_segment(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        segments.append(_finalize_segment(cur))
    return segments


def _finalize_segment(words: List[Dict]) -> Dict:
    return {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": " ".join(w["word"] for w in words).strip(),
        "words": words,
    }


def build_segments(raw_segments: List[Dict], min_dur: float = 1.0, max_dur: float = 15.0) -> List[Dict]:
    if not raw_segments:
        return []

    # Merge short segments
    merged = []
    buf = raw_segments[0].copy()
    for seg in raw_segments[1:]:
        if buf["end"] - buf["start"] < min_dur:
            buf["end"] = seg["end"]
            buf["text"] = (buf["text"] + " " + seg["text"]).strip()
            buf["words"] = buf.get("words", []) + seg.get("words", [])
        else:
            merged.append(buf)
            buf = seg.copy()
    merged.append(buf)

    # Split long segments recursively at the biggest pause gap
    def _split_seg(seg):
        dur = seg["end"] - seg["start"]
        words = seg.get("words", [])
        if dur > max_dur and len(words) >= 4:
            best_idx, best_gap = -1, 0.0
            for i in range(1, len(words)):
                g = words[i]["start"] - words[i - 1]["end"]
                if g > best_gap:
                    best_gap, best_idx = g, i
            if best_idx > 0 and best_gap > 0.15:
                lw, rw = words[:best_idx], words[best_idx:]
                left = {"start": seg["start"], "end": lw[-1]["end"],
                        "text": " ".join(w["word"] for w in lw).strip(), "words": lw}
                right = {"start": rw[0]["start"], "end": seg["end"],
                         "text": " ".join(w["word"] for w in rw).strip(), "words": rw}
                return _split_seg(left) + _split_seg(right)
        return [seg]

    final = []
    for seg in merged:
        final.extend(_split_seg(seg))

    output = []
    for seg in final:
        words = seg.get("words", [])
        s = words[0]["start"] if words else seg["start"]
        e = words[-1]["end"] if words else seg["end"]
        output.append({
            "start": s,
            "end": e,
            "duration": e - s,
            "text": seg.get("text", ""),
            "words": words,
            "word_count": len(words),
        })
    return output


def score_segments(
    segments: List[Dict],
    total_duration: float,
    keywords: List[str] = None,
    keyword_boost: float = 1.5,
) -> List[Dict]:
    scored = []
    for idx, seg in enumerate(segments):
        words = seg.get("words", [])
        dur = max(seg["duration"], 1e-6)

        # Confidence
        confidence = (sum(float(w.get("probability", 0.8)) for w in words) / len(words)) if words else 0.6

        # Density (words per second, normalised to ~3 wps)
        density = min(len(words) / dur / 3.0, 1.0)

        # Keyword hits
        kw_hits = 0
        kw_score = 0.0
        if keywords:
            lower = seg.get("text", "").lower()
            kw_hits = sum(lower.count(k.lower()) for k in keywords)
            kw_score = min(kw_hits * keyword_boost / 3.0, 1.0)

        # Position bonus (favour first 15% and last 10%)
        mid = (seg["start"] + seg["end"]) / 2.0
        rel = mid / max(total_duration, 1e-6)
        position = 1.0 if rel <= 0.15 else (0.8 if rel >= 0.90 else 0.4)

        score = 0.35 * confidence + 0.25 * density + 0.25 * kw_score + 0.15 * position

        scored.append({
            **seg,
            "score": score,
            "confidence": confidence,
            "density_wps": len(words) / dur,
            "keyword_hits": kw_hits,
        })
    return scored


# ---------------------------------------------------------------------------
# OpenAI Whisper transcription
# ---------------------------------------------------------------------------

MAX_BYTES = 24 * 1024 * 1024   # 24MB — stay under OpenAI's 25MB limit
CHUNK_SECS = 600               # 10-minute chunks for long files
CHUNK_OVERLAP = 5              # 5s overlap between chunks to avoid boundary gaps


def _extract_audio(video_path: str, out_path: str,
                   ss: float = 0.0, duration: float = None) -> None:
    """Extract mono 64kbps mp3 from video using ffmpeg."""
    import subprocess
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn",
           "-acodec", "libmp3lame", "-ab", "64k", "-ac", "1", "-ar", "16000"]
    if ss > 0:
        cmd += ["-ss", str(ss)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd.append(out_path)
    subprocess.run(cmd, capture_output=True, check=True)


def _transcribe_chunk(client, audio_path: str, offset: float = 0.0) -> List[Dict]:
    """Call OpenAI Whisper API on an audio file, return words with timestamps offset."""
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word"],
            language="en",
        )
    words = []
    for w in getattr(response, "words", None) or []:
        words.append({
            "word": w.word,
            "start": float(w.start) + offset,
            "end": float(w.end) + offset,
            "probability": 0.9,  # OpenAI doesn't expose per-word confidence
        })
    return words


def transcribe(video_path: str) -> Dict:
    import tempfile
    try:
        from openai import OpenAI
    except ImportError:
        return {"error": "openai not installed. Run: pip install openai"}

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY environment variable not set"}

    base_url = os.environ.get("OPENAI_BASE_URL", "https://proxy.shopify.ai/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)

    with tempfile.TemporaryDirectory() as tmp:
        # Extract full audio first to check size
        full_mp3 = f"{tmp}/full.mp3"
        print("Extracting audio...", file=sys.stderr)
        _extract_audio(video_path, full_mp3)
        size = os.path.getsize(full_mp3)
        print(f"Audio size: {size / 1024 / 1024:.1f}MB", file=sys.stderr)

        all_words: List[Dict] = []

        if size <= MAX_BYTES:
            # Single API call
            print("Transcribing via OpenAI Whisper...", file=sys.stderr)
            all_words = _transcribe_chunk(client, full_mp3, offset=0.0)
        else:
            # Chunk into 10-minute pieces with 5s overlap
            import subprocess, json as _json
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", video_path],
                capture_output=True, text=True)
            total = float(_json.loads(probe.stdout)["format"]["duration"])
            starts = list(range(0, int(total), CHUNK_SECS - CHUNK_OVERLAP))
            print(f"File too large — splitting into {len(starts)} chunks...", file=sys.stderr)
            seen_ends: set = set()
            for i, start in enumerate(starts):
                chunk_mp3 = f"{tmp}/chunk_{i}.mp3"
                dur = min(CHUNK_SECS, total - start + CHUNK_OVERLAP)
                _extract_audio(video_path, chunk_mp3, ss=start, duration=dur)
                print(f"  Transcribing chunk {i+1}/{len(starts)} "
                      f"({start:.0f}s–{start+dur:.0f}s)...", file=sys.stderr)
                chunk_words = _transcribe_chunk(client, chunk_mp3, offset=start)
                # Deduplicate overlap: skip words whose rounded end was already seen
                for w in chunk_words:
                    key = round(w["end"], 1)
                    if key not in seen_ends:
                        all_words.append(w)
                        seen_ends.add(key)

    if not all_words:
        return {"error": "No speech detected"}

    all_words = _pad_words(all_words)
    raw = list(_words_to_segments(all_words))
    segments = build_segments(raw)
    total_dur = max(w["end"] for w in all_words)
    segments = score_segments(segments, total_dur)

    # Tag every segment and word with the absolute source path
    abs_path = str(Path(video_path).resolve())
    for seg in segments:
        seg["source_video"] = abs_path
    for w in all_words:
        w["source_video"] = abs_path

    return {
        "segments": segments,
        "words": all_words,
        "total_duration": total_dur,
        "method": "openai-whisper",
        "sources": [abs_path],
    }


def transcribe_many(video_paths: List[str], no_cache: bool = False) -> Dict:
    """Transcribe multiple videos and merge into a single combined transcript.

    Each segment/word is tagged with source_video so downstream tools can map
    segments back to the correct source file when building a multi-clip timeline.
    """
    all_segments: List[Dict] = []
    all_words: List[Dict] = []
    sources: List[str] = []

    for path in video_paths:
        abs_path = str(Path(path).resolve())
        sources.append(abs_path)

        if not no_cache:
            cached = load_cached(abs_path)
            if cached:
                print(f"Using cached transcript for {Path(abs_path).name} "
                      f"({len(cached['segments'])} segments)", file=sys.stderr)
                # Back-fill source_video on old caches that predate this field
                for seg in cached.get("segments", []):
                    seg.setdefault("source_video", abs_path)
                for w in cached.get("words", []):
                    w.setdefault("source_video", abs_path)
                all_segments.extend(cached["segments"])
                all_words.extend(cached["words"])
                continue

        print(f"Transcribing: {Path(abs_path).name}", file=sys.stderr)
        result = transcribe(abs_path)
        if "error" in result:
            print(f"Error transcribing {Path(abs_path).name}: {result['error']}", file=sys.stderr)
            continue

        save_cache(abs_path, result)
        print(f"Cached: {_cache_path(abs_path)}", file=sys.stderr)
        all_segments.extend(result["segments"])
        all_words.extend(result["words"])

    if not all_words:
        return {"error": "No speech detected in any source file"}

    total_dur = max(w["end"] for w in all_words)
    return {
        "segments": all_segments,
        "words": all_words,
        "total_duration": total_dur,
        "method": "openai-whisper",
        "sources": sources,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe one or more videos with word-accurate timestamps.")
    parser.add_argument("video_paths", nargs="+", help="Path(s) to source video file(s)")
    parser.add_argument("--no-cache", action="store_true", help="Skip cache lookup and force re-transcription")
    args = parser.parse_args()

    video_paths = expand_paths(args.video_paths)

    if not video_paths:
        print("ERROR: No video files found.", file=sys.stderr)
        sys.exit(1)

    for path in video_paths:
        if not Path(path).exists():
            print(f"File not found: {path}", file=sys.stderr)
            sys.exit(1)

    if len(video_paths) == 1:
        path = video_paths[0]
        # Check cache first
        if not args.no_cache:
            cached = load_cached(path)
            if cached:
                print(f"Using cached transcript ({len(cached['segments'])} segments)", file=sys.stderr)
                # Back-fill source_video on old caches
                abs_path = str(Path(path).resolve())
                for seg in cached.get("segments", []):
                    seg.setdefault("source_video", abs_path)
                for w in cached.get("words", []):
                    w.setdefault("source_video", abs_path)
                json.dump(cached, sys.stdout, indent=2)
                print()
                sys.exit(0)

        print(f"Transcribing: {path}", file=sys.stderr)
        result = transcribe(path)

        if "error" in result:
            print(f"Error: {result['error']}", file=sys.stderr)
            sys.exit(1)

        seg_count = len(result["segments"])
        dur = result["total_duration"]
        print(f"Done: {seg_count} segments, {dur:.1f}s total", file=sys.stderr)

        save_cache(path, result)
        print(f"Cached to: {_cache_path(path)}", file=sys.stderr)
    else:
        # Multiple files: use transcribe_many()
        result = transcribe_many(video_paths, no_cache=args.no_cache)
        if "error" in result:
            print(f"Error: {result['error']}", file=sys.stderr)
            sys.exit(1)
        seg_count = len(result["segments"])
        dur = result["total_duration"]
        n = len(result.get("sources", []))
        print(f"Done: {seg_count} segments from {n} files, {dur:.1f}s total", file=sys.stderr)

    json.dump(result, sys.stdout, indent=2)
    print()  # trailing newline
