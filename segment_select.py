#!/usr/bin/env python3
"""
select.py — GPT-4o powered segment selection for resolve-autocut.

Reads a transcript JSON (from transcribe.py), asks GPT-4o to:
  - Find the best cold open (8-15s hook)
  - Select coherent segments covering the topic
  - Ensure clean in-points (no filler word starts)
  - Order segments for narrative flow

Outputs a segments.json ready for build_timeline.py.

Usage:
    python select.py transcript.json --topic "First Principles" --duration 180
    python select.py transcript.json --topic "future of e-commerce" --duration 180 > segments.json
    python select.py transcript.json --topic "product demo" --duration 120 --no-cold-open
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List


# Words/phrases that make for a poor in-point — GPT is asked to avoid these,
# but we also post-filter as a safety net.
FILLER_STARTS = (
    "and ", "but ", "so ", "i mean ", "um ", "uh ", "you know ",
    "like ", "right ", "yeah ", "ok ", "okay ", "well ", "now ",
)


def _has_filler_start(text: str) -> bool:
    lower = text.strip().lower()
    return any(lower.startswith(f) for f in FILLER_STARTS)


def _fmt_seg_list(segments: List[Dict]) -> str:
    lines = []
    for i, seg in enumerate(segments):
        dur = seg["end"] - seg["start"]
        filler = " [FILLER START]" if _has_filler_start(seg.get("text", "")) else ""
        lines.append(
            f"[{i}] {seg['start']:.1f}s–{seg['end']:.1f}s ({dur:.1f}s){filler}: {seg.get('text', '')}"
        )
    return "\n".join(lines)


def select_segments(
    transcript: Dict,
    topic: str,
    target_duration: float,
    cold_open: bool = True,
) -> Dict:
    """Use GPT-4o to select coherent segments from a transcript."""
    try:
        from openai import OpenAI
    except ImportError:
        return {"error": "openai not installed. Run: pip install openai"}

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY environment variable not set"}

    base_url = os.environ.get("OPENAI_BASE_URL", "https://proxy.shopify.ai/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)

    segments = transcript.get("segments", [])
    if not segments:
        return {"error": "No segments in transcript"}

    total_dur = transcript.get("total_duration", 0)
    seg_text = _fmt_seg_list(segments)

    cold_open_instruction = (
        "4. Identify the single best COLD OPEN: a compelling 8–15s moment that works as a "
        "hook — surprising, bold, or emotionally engaging. It does not need to be the "
        "chronological start of the video.\n"
        if cold_open else
        "4. cold_open_index: set to null (no cold open requested).\n"
    )

    system_prompt = """\
You are a master video editor and storyteller specializing in Shopify internal communications \
(all-hands meetings, town halls, leadership updates). You understand both the craft of video \
editing and Shopify's culture, voice, and mission deeply.

═══ SHOPIFY VOICE & CULTURE ═══
Shopify exists to give entrepreneurs the tools to achieve their dreams. The mission is always \
merchant-first. Ask yourself of every clip: does this ultimately connect to someone building \
their business? Clips about internal logistics or org structure only belong if they serve a \
larger point about merchant impact or company mission.

Shopify's communication style (Harley Finkelstein, Tobi Lütke, and the leadership team):
• Human and direct — conversational, not corporate. Contractions, plain language, no PR-speak.
• Meritocratic — entrepreneurship is the great equalizer. Results matter, not pedigree.
• Philosophical but grounded — willing to challenge assumptions, prefer meaning over metrics.
• Optimistic builders — "get in the arena." Celebrate wins that connect to real impact.
• Self-aware and sometimes self-deprecating — never takes itself too seriously.
• Simplicity-first — a one-sentence insight beats a five-minute explanation every time.

Shopify all-hands typically follow this narrative pattern:
  Hook → Bridge to mission → Evidence (wins/examples) → Meaning (why it matters) → Forward energy

═══ NARRATIVE STRUCTURE (every edit must have all three acts) ═══
1. HOOK (first 8–15s): Creates immediate curiosity, stakes, or delight. A concrete story, a \
surprising win, a bold claim. The viewer must feel "I need to keep watching." \
NO warm-up phrases, no "Hey everyone, welcome back," no context-setting openers.

2. BODY (middle sections): Evidence and substance. Clips alternate between concrete examples \
and broader significance. Energy varies — faster exposition clips, slower moments of meaning. \
Each clip builds on what came before.

3. PAYOFF (final 10–20s): Lands the emotional or strategic point — pride, mission, momentum. \
NOT a sign-off or general closing remarks. Those get cut or deferred to the very end.

═══ TRANSITION COHERENCE (most common failure point) ═══
Before finalizing the order, mentally "watch" each clip boundary. Ask:
• Does the FIRST SENTENCE of clip N+1 make sense immediately after the LAST SENTENCE of clip N?
• Are there DANGLING REFERENCES — phrases that assume context not established in previous clips?

DANGLING REFERENCE patterns to catch:
  - "Speaking of [X]..." — requires X to have been mentioned in a prior clip
  - "As I mentioned..." / "Going back to what I said..." — assumes prior context
  - "That's why..." / "Which is why..." — assumes a cause was already established
  - Opening with a pronoun ("He said...", "They did...", "It turns out...") — requires the \
    referent to be established
  - "And [continuing thought]..." or "But [countering previous]..." — assumes prior setup

Each clip must be SELF-CONTAINED enough that a viewer jumping in at that moment is not lost. \
Prefer clips that START with a new, complete idea rather than a callback.

═══ CUT POINT QUALITY ═══
• GOOD start: New thought, concrete noun, action, or question. Example: "Last Wednesday, the \
  New York Times crossword had a clue..."
• BAD start: Dangling reference, filler opener, mid-sentence continuation
• GOOD end: Completed thought, punchline, natural breath, rhetorical question
• BAD end: Mid-sentence ("...and so we"), trailing conjunction, half-finished idea
• SIGN-OFFS ("have a great weekend", "that's it for me", "[Name] out", "I'm so proud of...") \
  belong ONLY as the absolute final clip. If a great clip ends with sign-off bleed, note it — \
  trim_pass.py will handle it."""

    user_prompt = f"""Create a {target_duration:.0f}-second highlight reel from this Shopify all-hands transcript.

FOCUS TOPIC: "{topic}"
TARGET DURATION: {target_duration:.0f}s (HARD minimum {target_duration * 0.9:.0f}s, maximum {target_duration * 1.1:.0f}s — you MUST reach the minimum, include good-but-not-perfect clips if needed)
TOTAL SOURCE DURATION: {total_dur:.0f}s

SELECTION RULES:
1. Segments must be relevant to the focus topic
2. Total duration must be within the acceptable range
3. Prefer segments that START cleanly — avoid [FILLER START] segments unless content is exceptional
{cold_open_instruction}5. ORDER segments for narrative arc: HOOK → BODY → PAYOFF — you may reorder from original timeline
6. Avoid redundant or repetitive content — pick the single best version of each story beat
7. Each clip must be self-contained enough to not confuse a viewer who missed prior clips
8. CHECK EVERY TRANSITION: After finalizing your order, verify clip N+1's opening sentence \
makes sense after clip N's closing sentence. Flag any dangling references. \
ALSO check clip 0 (the very first clip): does it work as a standalone opening? A cold open \
that starts "Speaking of X..." or "As I was saying..." fails even with no prior clip.
9. MAP narrative roles: assign each clip to HOOK, SETUP, BODY, or PAYOFF. \
No two consecutive clips should serve the same role (except BODY clips).
10. SHOPIFY MISSION TEST: Does each clip ultimately connect to helping entrepreneurs or merchant \
impact? Pure internal logistics clips are excluded unless they serve a larger point.

SEGMENTS:
{seg_text}

Respond with JSON only, no markdown:
{{
  "cold_open_index": <integer segment index, or null>,
  "selected_indices": [<integers in final playback order>],
  "total_duration_s": <sum of durations of selected segments>,
  "summary": "<2-3 sentence description of what you selected and the narrative arc>",
  "flow_note": "<brief note on why this ordering creates a coherent story>",
  "sign_off_bleed": "<segment index of any clip ending with sign-off content, or empty string>",
  "excluded_reason": "<brief note on high-quality segments you excluded and why>"
}}"""

    print("Selecting segments with GPT-4o...", file=sys.stderr)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    raw = response.choices[0].message.content
    try:
        gpt_result = json.loads(raw)
    except json.JSONDecodeError as e:
        return {"error": f"GPT-4o returned invalid JSON: {e}\n{raw[:500]}"}

    cold_open_idx = gpt_result.get("cold_open_index")
    selected_indices = gpt_result.get("selected_indices", [])

    # Validate indices
    selected_indices = [i for i in selected_indices if isinstance(i, int) and 0 <= i < len(segments)]

    # Promote cold open to position 0 (if not already there)
    if cold_open_idx is not None and cold_open_idx in selected_indices:
        selected_indices = [cold_open_idx] + [i for i in selected_indices if i != cold_open_idx]

    # Enforce duration window: trim from the end if over max, warn if under min
    max_dur = target_duration * 1.1
    min_dur = target_duration * 0.9
    total = sum(segments[i]["end"] - segments[i]["start"] for i in selected_indices)
    if total > max_dur:
        # Drop non-cold-open clips from the end until within range
        trimmed = list(selected_indices)
        while trimmed and sum(segments[i]["end"] - segments[i]["start"] for i in trimmed) > max_dur:
            # Never remove cold open (position 0)
            trimmed.pop()
        selected_indices = trimmed
    elif total < min_dur:
        print(f"  [warn] Selected {total:.0f}s < minimum {min_dur:.0f}s — GPT under-selected", file=sys.stderr)

    selected_segs = []
    for idx in selected_indices:
        seg = dict(segments[idx])
        seg["_orig_idx"] = idx  # preserve for narrative_map lookup
        if idx == cold_open_idx:
            seg["_cold_open"] = True
        selected_segs.append(seg)

    actual_duration = sum(s["end"] - s["start"] for s in selected_segs)

    return {
        "segments": selected_segs,
        "summary": gpt_result.get("summary", ""),
        "flow_note": gpt_result.get("flow_note", ""),
        "transition_notes": gpt_result.get("transition_notes", []),
        "narrative_map": gpt_result.get("narrative_map", []),
        "sign_off_bleed": gpt_result.get("sign_off_bleed", ""),
        "excluded_reason": gpt_result.get("excluded_reason", ""),
        "total_duration": actual_duration,
        "cold_open_index": cold_open_idx,
        "segment_count": len(selected_segs),
    }


def print_selection_report(result: Dict) -> None:
    """Print a human-readable summary to stderr."""
    print("\n" + "=" * 60, file=sys.stderr)
    print("SELECTION REPORT", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    if result.get("summary"):
        print(f"\nSummary: {result['summary']}", file=sys.stderr)
    if result.get("flow_note"):
        print(f"Flow: {result['flow_note']}", file=sys.stderr)

    segments = result.get("segments", [])
    total = result.get("total_duration", 0)
    print(f"\n{len(segments)} segments selected, {total:.1f}s total ({total/60:.1f} min)\n", file=sys.stderr)

    # Build narrative role lookup by original segment index
    narrative_map = result.get("narrative_map", [])
    role_by_orig_idx = {entry.get("index"): entry.get("role", "") for entry in narrative_map}

    transition_notes = result.get("transition_notes", [])

    for i, seg in enumerate(segments):
        label = " [COLD OPEN]" if seg.get("_cold_open") else ""
        dur = seg["end"] - seg["start"]
        orig_idx = seg.get("_orig_idx", i)
        role = role_by_orig_idx.get(orig_idx, "")
        role_label = f" [{role}]" if role and not seg.get("_cold_open") else ""
        text_preview = seg.get("text", "")[:80]
        if len(seg.get("text", "")) > 80:
            text_preview += "..."
        print(f"  {i+1:2d}.{label}{role_label} {seg['start']:.1f}s–{seg['end']:.1f}s ({dur:.1f}s)",
              file=sys.stderr)
        print(f"      \"{text_preview}\"", file=sys.stderr)

        # Show transition to next clip
        if i < len(segments) - 1 and transition_notes and i < len(transition_notes):
            note = transition_notes[i]
            flag = "  ⚠ " if "DANGLING" in note.upper() else "  → "
            print(f"      {flag}{note}", file=sys.stderr)

    if result.get("sign_off_bleed"):
        print(f"\n⚠ Sign-off bleed detected: {result['sign_off_bleed']}", file=sys.stderr)
        print(f"  Run --trim to split and defer sign-off content to end.", file=sys.stderr)

    if result.get("excluded_reason"):
        print(f"\nExcluded: {result['excluded_reason']}", file=sys.stderr)

    print("=" * 60 + "\n", file=sys.stderr)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GPT-4o segment selection for resolve-autocut."
    )
    parser.add_argument("transcript_json", help="Path to transcript JSON from transcribe.py")
    parser.add_argument("--topic", required=True, help="Focus topic / keywords for selection")
    parser.add_argument("--duration", type=float, required=True,
                        help="Target duration in seconds (e.g. 180 for 3 minutes)")
    parser.add_argument("--no-cold-open", action="store_true",
                        help="Disable cold open detection")
    parser.add_argument("--output", "-o", default=None,
                        help="Write segments JSON to this file (default: stdout)")
    args = parser.parse_args()

    transcript_path = Path(args.transcript_json)
    if not transcript_path.exists():
        print(f"File not found: {transcript_path}", file=sys.stderr)
        sys.exit(1)

    with open(transcript_path) as f:
        transcript = json.load(f)

    # Accept either {segments, words, ...} or direct list
    if isinstance(transcript, list):
        transcript = {"segments": transcript}

    result = select_segments(
        transcript,
        topic=args.topic,
        target_duration=args.duration,
        cold_open=not args.no_cold_open,
    )

    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print_selection_report(result)

    output_data = result["segments"]

    if args.output:
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"Saved {len(output_data)} segments to: {args.output}", file=sys.stderr)
    else:
        json.dump(output_data, sys.stdout, indent=2)
        print()
