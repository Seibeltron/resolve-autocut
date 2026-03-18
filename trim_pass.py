#!/usr/bin/env python3
"""
trim_pass.py — GPT-4o word-level trim pass for resolve-autocut.

Takes selected segments (from segment_select.py) and tightens each clip's
in/out points using word-level timestamps from the transcript. GPT-4o can:
  - Trim a bad opener (filler words, mid-sentence start)
  - Trim a bad closer (sign-off bleed, closing remarks)
  - Split a segment into two (e.g. story + outro as separate clips)
  - Reorder split parts (e.g. move outro to end)

Usage:
    python trim_pass.py segments.json transcript.json
    python trim_pass.py segments.json transcript.json -o segments_trimmed.json
    python trim_pass.py /tmp/hwh_segments.json /tmp/hwh_transcript.json -o /tmp/hwh_trimmed.json
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _words_for_segment(seg: Dict, all_words: List[Dict]) -> List[Dict]:
    """Return words from the transcript that fall within the segment's time range.

    When source_video is present on the segment, only words from the same source
    file are considered — this prevents false matches when two source files have
    overlapping timestamps (both starting near 0s).
    """
    source = seg.get("source_video")
    return [
        w for w in all_words
        if (source is None or w.get("source_video") == source)
        and w["start"] >= seg["start"] - 0.1 and w["end"] <= seg["end"] + 0.1
    ]


def _fmt_words(words: List[Dict]) -> str:
    lines = []
    for w in words:
        lines.append(f"  {w['start']:.2f}s: {w['word']}")
    return "\n".join(lines)


def trim_segments(
    segments: List[Dict],
    transcript: Dict,
    context_note: str = "",
    keep_phrases: List[str] = None,
) -> Dict:
    """Use GPT-4o to tighten in/out points on each segment."""
    try:
        from openai import OpenAI
    except ImportError:
        return {"error": "openai not installed. Run: pip install openai"}

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}

    base_url = os.environ.get("OPENAI_BASE_URL", "https://proxy.shopify.ai/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)

    all_words = transcript.get("words", [])
    if not all_words:
        # Fall back to extracting words from segments in the transcript
        for s in transcript.get("segments", []):
            all_words.extend(s.get("words", []))

    # Build segment descriptions with word timestamps
    seg_descriptions = []
    for i, seg in enumerate(segments):
        words = _words_for_segment(seg, all_words)
        word_text = _fmt_words(words) if words else "  (no word timestamps available)"
        seg_descriptions.append(
            f"SEGMENT {i} [{seg['start']:.2f}s–{seg['end']:.2f}s] ({seg['end']-seg['start']:.1f}s)\n"
            f"Text: {seg.get('text', '')}\n"
            f"Words:\n{word_text}"
        )

    seg_block = "\n\n".join(seg_descriptions)
    context_block = f"\nContext: {context_note}" if context_note else ""

    keep_block = ""
    if keep_phrases:
        quoted = ", ".join(f'"{p}"' for p in keep_phrases)
        keep_block = (
            f"\n\nPROTECTED PHRASES — do NOT trim these, even if they appear to be filler or sign-offs:\n"
            f"{quoted}\n"
            "These are intentional payoff moments chosen by the editor."
        )

    system_prompt = (
        "You are a precise video editor specializing in Shopify internal communications "
        "(all-hands meetings, town halls, leadership updates). You receive selected segments "
        "with word-level timestamps and must tighten each clip's in/out points.\n\n"
        "WHAT TO TRIM:\n"
        "- Bad openers: filler words (um, uh, like, you know, right), mid-sentence starts, "
        "warm-up phrases ('hey everyone', 'so...', 'okay so')\n"
        "- Bad closers: sign-off bleed, closing remarks, trailing filler\n"
        "- Bridge phrases at the start ('Speaking of X', 'As I mentioned', 'Going back to...') "
        "if the content that follows is self-contained without that bridge\n\n"
        "WHAT TO PROTECT:\n"
        "- Punchlines and payoff moments — a callback or punchline at the END of a story "
        "('...and let's hope you got 13 across right') is the PAYOFF, not trailing filler. "
        "Only trim a closer if it is generic warm-up, not if it delivers the point of the story.\n"
        "- Concrete examples and merchant impact stories — never trim the part that connects "
        "to a real business or entrepreneurship outcome\n"
        "- The moment the speaker states the 'so what' — the meaning behind the anecdote\n\n"
        "SIGN-OFF HANDLING:\n"
        "Phrases like 'have a great weekend', '[name] out', 'that's it for me', 'I'm so proud of...' "
        "are sign-off bleed. If they appear at the END of a segment that also contains good content, "
        "use split with part_b_reorder='end' to defer the sign-off to the end of the edit. "
        "If the ENTIRE segment is a sign-off, leave it as-is (it may be the intentional final clip).\n\n"
        "BRIDGE PHRASE HANDLING:\n"
        "If a clip starts with 'Speaking of X' or similar bridge, trim the bridge phrase "
        "ONLY if what follows is self-contained. If the clip depends on unestablished context, "
        "note this in trim_note so the editor can decide whether to drop the clip.\n\n"
        "Use the word timestamps to identify exact cut points. New start/end must align to word "
        f"boundaries (use the word's start time for new_start, the word's end time for new_end)."
        f"{keep_block}"
    )

    user_prompt = f"""Tighten these {len(segments)} segments. For each segment, identify the tightest "
compelling in/out points and flag any content to split or reposition.{context_block}

{seg_block}

Respond with JSON only, no markdown:
{{
  "segments": [
    {{
      "original_index": <int>,
      "new_start": <float — word-aligned start timestamp>,
      "new_end": <float — word-aligned end timestamp>,
      "trim_note": "<what was trimmed and why, or 'no change'>",
      "split": {{
        "enabled": <bool — true if this segment should be split into 2 clips>,
        "split_at": <float — timestamp where split occurs, or null>,
        "part_b_reorder": <"end" | "after:<index>" | null — where to place part B>
      }}
    }}
  ],
  "overall_note": "<brief note on the overall trim decisions>"
}}"""

    print("Trimming segments with GPT-4o...", file=sys.stderr)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    raw = response.choices[0].message.content
    try:
        gpt_result = json.loads(raw)
    except json.JSONDecodeError as e:
        return {"error": f"GPT-4o returned invalid JSON: {e}\n{raw[:500]}"}

    # Apply trim decisions
    trimmed_segs = []
    deferred = []  # segments to place at end or after specific index

    for decision in gpt_result.get("segments", []):
        orig_idx = decision.get("original_index", 0)
        if orig_idx >= len(segments):
            continue

        orig = dict(segments[orig_idx])
        new_start = decision.get("new_start", orig["start"])
        new_end = decision.get("new_end", orig["end"])
        trim_note = decision.get("trim_note", "no change")
        split_info = decision.get("split", {}) or {}

        if split_info.get("enabled") and split_info.get("split_at"):
            split_at = split_info["split_at"]
            part_a = dict(orig)
            part_a["start"] = new_start
            part_a["end"] = split_at
            part_a["trim_note"] = trim_note

            part_b = dict(orig)
            part_b["start"] = split_at
            part_b["end"] = new_end
            part_b["_split_from"] = orig_idx
            part_b["trim_note"] = "split from segment " + str(orig_idx)

            reorder = split_info.get("part_b_reorder")
            if reorder == "end":
                trimmed_segs.append(part_a)
                deferred.append(part_b)
            elif reorder and reorder.startswith("after:"):
                trimmed_segs.append(part_a)
                deferred.append((int(reorder.split(":")[1]), part_b))
            else:
                trimmed_segs.append(part_a)
                trimmed_segs.append(part_b)
        else:
            seg_out = dict(orig)
            seg_out["start"] = new_start
            seg_out["end"] = new_end
            seg_out["trim_note"] = trim_note
            trimmed_segs.append(seg_out)

        print(f"  Seg {orig_idx}: {trim_note}", file=sys.stderr)

    # Insert deferred segments
    for item in deferred:
        if isinstance(item, tuple):
            after_idx, seg = item
            insert_pos = after_idx + 1
            trimmed_segs.insert(min(insert_pos, len(trimmed_segs)), seg)
        else:
            trimmed_segs.append(item)

    # Drop zero-duration artifacts (e.g. split_at == new_end)
    trimmed_segs = [s for s in trimmed_segs if s["end"] - s["start"] > 0.05]

    total = sum(s["end"] - s["start"] for s in trimmed_segs)
    return {
        "segments": trimmed_segs,
        "overall_note": gpt_result.get("overall_note", ""),
        "total_duration": total,
    }


def print_trim_report(original: List[Dict], result: Dict) -> None:
    print("\n" + "=" * 60, file=sys.stderr)
    print("TRIM REPORT", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    if result.get("overall_note"):
        print(f"\n{result['overall_note']}\n", file=sys.stderr)

    orig_total = sum(s["end"] - s["start"] for s in original)
    new_total = result.get("total_duration", 0)
    trimmed = original_count = len(original)
    new_count = len(result.get("segments", []))

    print(f"Segments: {original_count} → {new_count}", file=sys.stderr)
    print(f"Duration: {orig_total:.1f}s → {new_total:.1f}s ({orig_total - new_total:+.1f}s)\n",
          file=sys.stderr)

    for i, seg in enumerate(result.get("segments", [])):
        dur = seg["end"] - seg["start"]
        label = " [split]" if "_split_from" in seg else ""
        print(f"  {i+1:2d}.{label} {seg['start']:.1f}s–{seg['end']:.1f}s ({dur:.1f}s)",
              file=sys.stderr)
        preview = seg.get("text", "")[:70]
        if preview:
            print(f"      \"{preview}\"", file=sys.stderr)

    print("=" * 60 + "\n", file=sys.stderr)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GPT-4o word-level trim pass for resolve-autocut segments."
    )
    parser.add_argument("segments_json", help="Segments JSON from segment_select.py")
    parser.add_argument("transcript_json", help="Transcript JSON from transcribe.py")
    parser.add_argument("--context", default="", help="Optional context note for GPT (e.g. 'move outro to end')")
    parser.add_argument("--keep", action="append", metavar="PHRASE", default=[],
                        help="Protect a phrase from being trimmed (can use multiple times)")
    parser.add_argument("--output", "-o", default=None,
                        help="Write trimmed segments JSON to this file (default: stdout)")
    args = parser.parse_args()

    for p in [args.segments_json, args.transcript_json]:
        if not Path(p).exists():
            print(f"File not found: {p}", file=sys.stderr)
            sys.exit(1)

    with open(args.segments_json) as f:
        segments = json.load(f)
    if isinstance(segments, dict) and "segments" in segments:
        segments = segments["segments"]

    with open(args.transcript_json) as f:
        transcript = json.load(f)

    result = trim_segments(segments, transcript, context_note=args.context,
                           keep_phrases=args.keep or None)

    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print_trim_report(segments, result)

    output_data = result["segments"]
    if args.output:
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"Saved {len(output_data)} trimmed segments to: {args.output}", file=sys.stderr)
    else:
        json.dump(output_data, sys.stdout, indent=2)
        print()
