#!/usr/bin/env python3
"""
build_timeline.py — Build a non-destructive DaVinci Resolve timeline from selected segments.

Each segment becomes a separate clip item on the timeline referencing the original source
file with in/out points. All cut points remain adjustable in Resolve's Edit page.

Usage:
    python build_timeline.py /path/to/video.mp4 /path/to/segments.json
    python build_timeline.py /path/to/video.mp4 /path/to/segments.json --timeline-name "My Cut"
    python build_timeline.py /path/to/segments.json   (multi-source: source_video embedded in segments)
"""

import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def setup_resolve_env():
    """Set up environment variables and Python path for Resolve scripting API."""
    import platform
    if platform.system() == "Darwin":
        api_path = "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting"
        lib_path = "/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so"
    elif platform.system() == "Windows":
        api_path = os.path.join(os.environ.get("PROGRAMDATA", "C:\\ProgramData"),
                                "Blackmagic Design", "DaVinci Resolve", "Support", "Developer", "Scripting")
        lib_path = os.path.join(os.environ.get("PROGRAMFILES", "C:\\Program Files"),
                                "Blackmagic Design", "DaVinci Resolve", "fusionscript.dll")
    else:  # Linux
        api_path = "/opt/resolve/Developer/Scripting"
        lib_path = "/opt/resolve/libs/fusionscript.so"

    modules_path = os.path.join(api_path, "Modules")
    os.environ["RESOLVE_SCRIPT_API"] = api_path
    os.environ["RESOLVE_SCRIPT_LIB"] = lib_path
    if modules_path not in sys.path:
        sys.path.insert(0, modules_path)


def connect_to_resolve():
    setup_resolve_env()
    try:
        import DaVinciResolveScript as dvr
        resolve = dvr.scriptapp("Resolve")
        if resolve is None:
            print("ERROR: Could not connect to DaVinci Resolve. Make sure Resolve is running.", file=sys.stderr)
            sys.exit(1)
        return resolve
    except ImportError as e:
        print(f"ERROR: Could not import DaVinci Resolve scripting module: {e}", file=sys.stderr)
        sys.exit(1)


def get_source_fps(video_path: str) -> float:
    """Read the source file's native frame rate using ffprobe."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(out.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                r = stream.get("r_frame_rate", "")
                if "/" in r:
                    num, den = r.split("/")
                    return float(Fraction(int(num), int(den)))
    except Exception as e:
        print(f"Warning: ffprobe failed ({e}), will use clip property", file=sys.stderr)
    return 0.0


# Map exact fps fractions to Resolve's timeline setting strings
_FPS_SETTING = {
    23.976: "23.976", 24.0: "24", 25.0: "25",
    29.97: "29.97",  30.0: "30", 47.952: "47.952",
    48.0: "48",      50.0: "50", 59.94: "59.94",
    60.0: "60",
}


def _get_openai_client():
    """Return an OpenAI client using env vars, or None if unavailable."""
    try:
        from openai import OpenAI
    except ImportError:
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    base_url = os.environ.get("OPENAI_BASE_URL", "https://proxy.shopify.ai/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def _refine_one_clip(args: tuple) -> tuple:
    """Transcribe the window around a single clip's cut point and determine if extension needed.
    Designed to run in a thread pool.

    Returns: (index, updated_entry_dict, status_message)
    """
    i, entry, next_start_frame, video_path, fps, min_slack_s, target_slack_s, client = args

    cut_s = entry["endFrame"] / fps
    look_back_s = 1.0
    look_ahead_s = 4.0
    win_start = max(0.0, cut_s - look_back_s)
    win_end = cut_s + look_ahead_s

    import tempfile
    tmp_mp3 = None
    words = []
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_mp3 = f.name
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{win_start:.6f}", "-to", f"{win_end:.6f}",
             "-i", video_path, "-vn", "-acodec", "libmp3lame", "-ab", "64k",
             "-ac", "1", "-ar", "16000", tmp_mp3],
            capture_output=True, timeout=30,
        )
        with open(tmp_mp3, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["word"],
                language="en",
            )
        for w in getattr(response, "words", None) or []:
            words.append({
                "word": w.word,
                "start": float(w.start) + win_start,
                "end": float(w.end) + win_start,
            })
    except Exception as e:
        return i, dict(entry), f"⚠ error: {e}"
    finally:
        if tmp_mp3:
            try:
                os.unlink(tmp_mp3)
            except OSError:
                pass

    # Determine if this cut needs extending
    straddling = [w for w in words if w["start"] < cut_s < w["end"]]
    close_after = [w for w in words if cut_s <= w["end"] <= cut_s + 0.4]
    before = [w for w in words if w["end"] <= cut_s]
    slack = (cut_s - before[-1]["end"]) if before else 999.0

    needs_extend = bool(straddling or close_after) or slack < min_slack_s
    entry = dict(entry)

    if needs_extend:
        if straddling or close_after:
            after_cut = sorted(
                [w for w in words if w["start"] >= cut_s - 0.1],
                key=lambda w: w["start"],
            )
            phrase_end_s = None
            for j in range(len(after_cut) - 1):
                if after_cut[j]["end"] > cut_s:
                    if after_cut[j + 1]["start"] - after_cut[j]["end"] > 0.3:
                        phrase_end_s = after_cut[j]["end"]
                        break
            if phrase_end_s is None and after_cut:
                phrase_end_s = max(w["end"] for w in after_cut)
            new_end_s = (phrase_end_s or cut_s) + target_slack_s
        else:
            new_end_s = cut_s + target_slack_s

        new_end_frame = int(new_end_s * fps)
        if next_start_frame is not None and next_start_frame > entry["endFrame"]:
            new_end_frame = min(new_end_frame, next_start_frame - 1)

        if new_end_frame > entry["endFrame"]:
            extend_ms = (new_end_frame - entry["endFrame"]) / fps * 1000
            if straddling or close_after:
                after_cut = sorted(
                    [w for w in words if w["start"] >= cut_s - 0.1],
                    key=lambda w: w["start"],
                )
                phrase_words = " ".join(
                    w["word"] for w in after_cut if w["end"] <= new_end_s
                ).strip()
                msg = f"⚠ extended +{extend_ms:.0f}ms (phrase: \"{phrase_words[-40:]}\" → pause)"
            else:
                msg = f"⚠ extended +{extend_ms:.0f}ms (slack {slack*1000:.0f}ms)"
            entry["endFrame"] = new_end_frame
        else:
            msg = "✓ OK"
    else:
        heard_tail = " ".join(w["word"] for w in before[-3:]).strip()
        msg = f"✓ OK (slack {slack*1000:.0f}ms, ends: \"{heard_tail}\")"

    return i, entry, msg


def refine_end_frames(clip_list: list, fps_map: dict,
                      min_slack_s: float = 0.15,
                      target_slack_s: float = 0.35) -> list:
    """Parallel refinement: for each clip, call OpenAI Whisper on the cut-point window
    and extend endFrame to the next natural phrase pause if a word is being cut off.
    Uses ThreadPoolExecutor for ~N-clip parallel API calls.

    fps_map: {source_video_path: fps} — used to look up per-clip fps.
    """
    client = _get_openai_client()
    if client is None:
        print("Warning: OpenAI not available (check OPENAI_API_KEY), skipping refinement.",
              file=sys.stderr)
        return clip_list

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Default fps for clips with no _source_video key (backwards compat)
    default_fps = next(iter(fps_map.values()), 24.0)
    default_source = next(iter(fps_map.keys()), "")

    # Build args for each clip — use per-clip source video and fps
    args_list = []
    for i, entry in enumerate(clip_list):
        source_video = entry.get("_source_video", default_source)
        fps = fps_map.get(source_video, default_fps)
        # Only use next clip's startFrame as a cap when it's from the SAME source file.
        # Cross-source frame numbers are unrelated and must not be used as a cap.
        if i + 1 < len(clip_list):
            next_entry = clip_list[i + 1]
            next_source = next_entry.get("_source_video", default_source)
            next_start = next_entry["startFrame"] if next_source == source_video else None
        else:
            next_start = None
        args_list.append((i, entry, next_start, source_video, fps, min_slack_s, target_slack_s, client))

    print(f"\nRefining {len(clip_list)} clips in parallel...", file=sys.stderr)
    results = [None] * len(clip_list)

    with ThreadPoolExecutor(max_workers=min(6, len(clip_list))) as pool:
        futures = {pool.submit(_refine_one_clip, a): a[0] for a in args_list}
        for future in as_completed(futures):
            i, entry, msg = future.result()
            results[i] = (entry, msg)

    print("\nRefinement report:", file=sys.stderr)
    refined = []
    for i, (entry, msg) in enumerate(results):
        print(f"  Clip {i+1}: {msg}", file=sys.stderr)
        refined.append(entry)

    return refined


def _import_or_find_clip(media_pool, video_path: str):
    """Import a video into the media pool, or return the existing clip if already there."""
    imported = media_pool.ImportMedia([video_path])
    if imported:
        print(f"  Imported: {Path(video_path).name}", file=sys.stderr)
        return imported[0]
    # Already in pool — search by filename
    video_name = Path(video_path).name
    root = media_pool.GetRootFolder()
    clip = next((c for c in (root.GetClipList() or []) if c.GetName() == video_name), None)
    if clip:
        print(f"  Found existing: {clip.GetName()}", file=sys.stderr)
        return clip
    print(f"  ERROR: Failed to import {video_name} and could not find it in media pool.", file=sys.stderr)
    return None


def build_timeline(video_path: str, segments: list, timeline_name: str = None,
                   refine: bool = True) -> bool:
    """Build a DaVinci Resolve timeline from selected segments.

    video_path: path to the source video (used for single-source or legacy runs).
                When segments contain a 'source_video' field, those paths are used
                instead and video_path may be None.
    """
    if not segments:
        print("ERROR: No segments provided.", file=sys.stderr)
        return False

    # --- Determine source video paths ---
    # If segments carry source_video, use those (multi-source or newer single-source runs).
    # Otherwise fall back to the explicit video_path arg (legacy).
    seg_sources = list(dict.fromkeys(
        s["source_video"] for s in segments if s.get("source_video")
    ))

    if seg_sources:
        video_paths = seg_sources
    else:
        if not video_path:
            print("ERROR: No video path provided and segments have no source_video field.", file=sys.stderr)
            return False
        video_paths = [str(Path(video_path).resolve())]
        # Tag all segments so downstream logic is uniform
        for seg in segments:
            seg["source_video"] = video_paths[0]

    for vp in video_paths:
        if not Path(vp).exists():
            print(f"ERROR: Video file not found: {vp}", file=sys.stderr)
            return False

    # --- Detect fps per source ---
    fps_map = {}
    for vp in video_paths:
        fps = get_source_fps(vp)
        if fps <= 0:
            fps = 24.0
            print(f"Warning: Could not detect fps for {Path(vp).name}, defaulting to {fps}", file=sys.stderr)
        fps_map[vp] = fps
        print(f"{Path(vp).name}: {fps} fps", file=sys.stderr)

    fps_values = set(fps_map.values())
    if len(fps_values) > 1:
        print(f"Warning: Source files have different frame rates: {sorted(fps_values)}", file=sys.stderr)
        print("Clips will use per-file fps for frame calculations. "
              "Resolve may reinterpret frame counts if the timeline fps differs from a source.", file=sys.stderr)

    # Primary fps for project validation (use first source)
    primary_fps = fps_map[video_paths[0]]
    fps_key = min(_FPS_SETTING, key=lambda k: abs(k - primary_fps))
    fps_setting = _FPS_SETTING[fps_key]

    print(f"", file=sys.stderr)
    print(f"IMPORTANT — Before proceeding, verify Resolve is set to {fps_setting} fps:", file=sys.stderr)
    print(f"  File > Project Settings > Master Settings", file=sys.stderr)
    print(f"    Timeline frame rate          → {fps_setting}", file=sys.stderr)
    print(f"    Playback frame rate          → {fps_setting}", file=sys.stderr)
    print(f"    Video monitoring format      → {fps_setting}", file=sys.stderr)
    print(f"", file=sys.stderr)

    print(f"Connecting to DaVinci Resolve...", file=sys.stderr)
    resolve = connect_to_resolve()

    project_manager = resolve.GetProjectManager()
    project = project_manager.GetCurrentProject()
    if project is None:
        print("ERROR: No project open in DaVinci Resolve. Please open a project first.", file=sys.stderr)
        return False

    print(f"Project: {project.GetName()}", file=sys.stderr)

    # --- Import all source videos into media pool ---
    media_pool = project.GetMediaPool()
    print(f"Importing {len(video_paths)} source file(s)...", file=sys.stderr)
    media_items = {}
    for vp in video_paths:
        clip = _import_or_find_clip(media_pool, vp)
        if clip is None:
            return False
        media_items[vp] = clip

    # --- Timeline name ---
    if timeline_name is None:
        if len(video_paths) == 1:
            timeline_name = Path(video_paths[0]).stem + "_autocut"
        else:
            timeline_name = Path(video_paths[0]).stem + "_multiclip_autocut"

    # Delete any existing timeline with this name to start fresh
    for i in range(project.GetTimelineCount(), 0, -1):
        existing = project.GetTimelineByIndex(i)
        if existing and existing.GetName() == timeline_name:
            media_pool.DeleteTimelines([existing])
            print(f"Deleted existing timeline '{timeline_name}'", file=sys.stderr)
            break

    # Verify project fps matches primary source; auto-set if possible
    project_fps = project.GetSetting("timelineFrameRate")
    if abs(float(project_fps) - float(fps_setting)) > 0.01:
        n_timelines = project.GetTimelineCount()
        auto_set_ok = False
        if n_timelines == 0:
            auto_set_ok = bool(project.SetSetting("timelineFrameRate", fps_setting))
        if auto_set_ok:
            print(f"Auto-set project fps to {fps_setting} ✓", file=sys.stderr)
        else:
            reason = (
                f"project has {n_timelines} existing timeline(s)"
                if n_timelines > 0
                else "SetSetting returned False"
            )
            print(f"\nERROR: Project fps ({project_fps}) does not match source fps ({fps_setting}).", file=sys.stderr)
            print(f"Could not auto-set fps ({reason}).", file=sys.stderr)
            print(f"Please set the timeline frame rate manually in Resolve:", file=sys.stderr)
            print(f"  File > Project Settings > Master Settings > Timeline frame rate > {fps_setting}", file=sys.stderr)
            print(f"Then re-run this script.", file=sys.stderr)
            return False
    else:
        print(f"Project fps matches source: {fps_setting} ✓", file=sys.stderr)

    # Create empty timeline
    print(f"Creating empty timeline '{timeline_name}'...", file=sys.stderr)
    timeline = media_pool.CreateEmptyTimeline(timeline_name)
    if not timeline:
        print(f"ERROR: Could not create timeline '{timeline_name}'", file=sys.stderr)
        return False

    project.SetCurrentTimeline(timeline)
    tl_fps = timeline.GetSetting("timelineFrameRate")
    print(f"Timeline created: {timeline_name} @ {tl_fps} fps", file=sys.stderr)

    # --- Merge adjacent same-source segments (gap < 0.5s) to avoid mid-sentence cuts ---
    merged = [dict(segments[0])]
    for seg in segments[1:]:
        prev = merged[-1]
        same_source = seg.get("source_video") == prev.get("source_video")
        gap = seg["start"] - prev["end"]
        if same_source and 0 <= gap < 0.5:
            prev["end"] = seg["end"]
            prev["text"] = (prev.get("text", "") + " " + seg.get("text", "")).strip()
        else:
            merged.append(dict(seg))
    if len(merged) < len(segments):
        print(f"Merged {len(segments)} segments → {len(merged)} clips "
              f"(removed {len(segments)-len(merged)} mid-sentence cuts)", file=sys.stderr)

    # --- Build clip list with per-source fps and mediaPoolItem ---
    end_pad_s = 0.25  # 250ms end pad to avoid clipping word endings
    clip_list = []
    for i, seg in enumerate(merged):
        seg_source = seg.get("source_video", video_paths[0])
        seg_fps = fps_map.get(seg_source, primary_fps)
        pool_item = media_items.get(seg_source, media_items[video_paths[0]])

        start_frame = int(seg["start"] * seg_fps)
        raw_end_frame = int(seg["end"] * seg_fps)
        end_frame = raw_end_frame + int(end_pad_s * seg_fps)

        # Cap end pad only when the next clip is from the same source and chronologically adjacent
        if i + 1 < len(merged):
            next_seg = merged[i + 1]
            next_source = next_seg.get("source_video", video_paths[0])
            if next_source == seg_source:
                next_start_frame = int(next_seg["start"] * seg_fps)
                if next_start_frame > raw_end_frame:
                    end_frame = min(end_frame, next_start_frame - 1)

        if end_frame <= start_frame:
            end_frame = start_frame + 1

        clip_list.append({
            "mediaPoolItem": pool_item,
            "startFrame": start_frame,
            "endFrame": end_frame,
            "_source_video": seg_source,  # stripped before AppendToTimeline
        })

    # Auto-skip refine if segments came from trim_pass (already word-boundary aligned)
    has_trim_notes = any("trim_note" in seg for seg in segments)
    if has_trim_notes and refine:
        print("Segments from trim_pass detected — skipping refine pass (already word-aligned).", file=sys.stderr)
        refine = False

    if refine:
        print(f"Refining end frames ({len(clip_list)} clips)...", file=sys.stderr)
        clip_list = refine_end_frames(clip_list, fps_map)

    # Strip internal keys before passing to Resolve API
    resolve_clip_list = [
        {k: v for k, v in entry.items() if not k.startswith("_")}
        for entry in clip_list
    ]

    print(f"Appending {len(resolve_clip_list)} clips to timeline...", file=sys.stderr)
    result = media_pool.AppendToTimeline(resolve_clip_list)
    if not result:
        print("ERROR: AppendToTimeline failed.", file=sys.stderr)
        return False

    total_dur = sum(seg["end"] - seg["start"] for seg in segments)
    src_summary = (
        f"{len(video_paths)} source files" if len(video_paths) > 1
        else Path(video_paths[0]).name
    )
    print(f"Done! Timeline '{timeline_name}' built with {len(resolve_clip_list)} clips "
          f"({total_dur:.1f}s total, from {src_summary})", file=sys.stderr)
    print(f"Switch to the Edit page in DaVinci Resolve to review your cut.", file=sys.stderr)

    resolve.OpenPage("edit")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a DaVinci Resolve timeline from selected segments.",
        epilog=(
            "Single-source: build_timeline.py video.mp4 segments.json\n"
            "Multi-source:  build_timeline.py segments.json  (source_video embedded in segments)"
        ),
    )
    # video_path is optional — multi-source runs derive paths from segments' source_video field
    parser.add_argument("video_path", nargs="?", default=None,
                        help="Path to the source video (omit when segments contain source_video)")
    parser.add_argument("segments_json", help="Path to JSON file with selected segments")
    parser.add_argument("--timeline-name", default=None, help="Name for the new timeline (default: video_stem_autocut)")
    parser.add_argument("--no-refine", action="store_true", help="Skip end-frame refinement pass (faster)")
    args = parser.parse_args()

    # If the first positional arg looks like a JSON file, the user omitted video_path
    if args.video_path and args.video_path.endswith(".json"):
        # Shift: video_path slot actually holds the segments file
        args.segments_json = args.video_path
        args.video_path = None

    segments_path = Path(args.segments_json)
    if not segments_path.exists():
        print(f"ERROR: segments file not found: {segments_path}", file=sys.stderr)
        sys.exit(1)

    with open(segments_path) as f:
        data = json.load(f)

    # Accept either a list of segments directly, or a dict with a "segments" key
    if isinstance(data, list):
        segments = data
    elif isinstance(data, dict) and "segments" in data:
        segments = data["segments"]
    else:
        print("ERROR: segments_json must be a list of segments or {\"segments\": [...]}", file=sys.stderr)
        sys.exit(1)

    success = build_timeline(args.video_path, segments, args.timeline_name,
                             refine=not args.no_refine)
    sys.exit(0 if success else 1)
