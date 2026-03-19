#!/usr/bin/env python3
"""
select.py — AI-powered segment selection for resolve-autocut.

Reads a transcript JSON (from transcribe.py), asks a configurable model to:
  - Find the best cold open (8-15s hook)
  - Select coherent segments covering the topic
  - Ensure clean in-points (no filler word starts)
  - Order segments for narrative flow

Outputs a segments.json ready for build_timeline.py.

Usage:
    python select.py transcript.json --topic "First Principles" --duration 180
    python select.py transcript.json --topic "future of e-commerce" --duration 180 > segments.json
    python select.py transcript.json --topic "product demo" --duration 120 --no-cold-open
    python select.py transcript.json --topic "product demo" --duration 120 --model anthropic:claude-sonnet-4-6
    python select.py transcript.json --topic "product demo" --duration 120 --model google:gemini-2.5-flash

Model examples (via Shopify proxy):
    gpt-4o              (default)
    gpt-4.1             newer OpenAI, often faster
    gpt-4.1-mini        cheaper, good for shorter transcripts
    anthropic:claude-sonnet-4-6
    anthropic:claude-sonnet-4-5
    anthropic:claude-haiku-4-5   fastest/cheapest Anthropic
    google:gemini-2.5-flash      large context window
    google:gemini-2.5-pro
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _parse_json_response(raw: str) -> dict:
    """Parse a JSON response that may be wrapped in markdown code fences."""
    clean = raw.strip()
    if clean.startswith("```"):
        # Strip opening fence (```json or ```)
        clean = clean.split("```", 2)[1]
        if clean.startswith("json"):
            clean = clean[4:]
        # Strip closing fence
        clean = clean.rsplit("```", 1)[0].strip()
    return json.loads(clean)


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
        source = seg.get("source_video", "")
        source_label = f" [SRC:{Path(source).name}]" if source else ""
        lines.append(
            f"[{i}] {seg['start']:.1f}s–{seg['end']:.1f}s ({dur:.1f}s){source_label}{filler}: {seg.get('text', '')}"
        )
    return "\n".join(lines)


def _is_multi_source(segments: List[Dict]) -> bool:
    sources = {s.get("source_video") for s in segments if s.get("source_video")}
    return len(sources) > 1


def suggest_topics(transcript: Dict, n: int = 6, model: str = "anthropic:claude-sonnet-4-6") -> List[str]:
    """Use an LLM to suggest N distinct topics present in the transcript."""
    try:
        from openai import OpenAI
    except ImportError:
        return []

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return []

    base_url = os.environ.get("OPENAI_BASE_URL", "https://proxy.shopify.ai/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)

    segments = transcript.get("segments", [])
    total_dur = transcript.get("total_duration", 0)

    # Sample down to ≤400 segments to stay within token limits for large multi-video transcripts
    MAX_SEGS = 400
    if len(segments) > MAX_SEGS:
        step = len(segments) / MAX_SEGS
        sampled = [segments[int(i * step)] for i in range(MAX_SEGS)]
        sample_note = f" (sampled {MAX_SEGS} of {len(segments)} segments)"
    else:
        sampled = segments
        sample_note = ""

    seg_text = _fmt_seg_list(sampled)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a video editor analyzing interview/testimonial transcripts. "
                    "Your job is to identify distinct, meaningful themes suitable for a highlight reel."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Here are {len(sampled)} transcript segments "
                    f"({total_dur:.0f}s total{sample_note}):\n\n"
                    f"{seg_text}\n\n"
                    f"Identify exactly {n} distinct topics or themes that appear in these segments "
                    "and would make compelling highlight reels. Each topic should be specific enough "
                    "to guide segment selection, not generic. "
                    "Return JSON only: {\"topics\": [\"topic 1\", \"topic 2\", ...]}"
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.4,
    )

    raw = response.choices[0].message.content
    try:
        return _parse_json_response(raw).get("topics", [])[:n]
    except (json.JSONDecodeError, AttributeError):
        return []


def prescore_segments(
    transcript: Dict,
    topic: str,
    threshold: float = 4.0,
    model: str = "gpt-4.1-mini",
) -> Dict:
    """Score each segment 1–10 with a cheap model; return transcript filtered to threshold+.

    Removes low-scoring segments before the full selection pass, reducing input size
    and steering the expensive model toward higher-quality material.
    Fails open — if scoring fails for any reason, the original transcript is returned.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return transcript

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return transcript

    base_url = os.environ.get("OPENAI_BASE_URL", "https://proxy.shopify.ai/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)

    segments = transcript.get("segments", [])
    if not segments:
        return transcript

    seg_text = _fmt_seg_list(segments)
    print(f"Pre-scoring {len(segments)} segments with {model}...", file=sys.stderr)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a video editor scoring transcript segments for editorial value. "
                    "Score each segment 1–10 based on: relevance to the topic, speech clarity, "
                    "whether it contains a complete thought, and narrative/emotional value. "
                    "1–3 = poor (off-topic, pure filler, incomplete sentence). "
                    "4–6 = usable but not compelling. "
                    "7–10 = strong editorial material."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Score each segment for a highlight reel focused on: \"{topic}\"\n\n"
                    f"Segments:\n{seg_text}\n\n"
                    "Return JSON only: {\"scores\": [score_for_seg_0, score_for_seg_1, ...]}\n"
                    "One integer score (1–10) per segment, in index order."
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    raw = response.choices[0].message.content
    try:
        scores = _parse_json_response(raw).get("scores", [])
    except (json.JSONDecodeError, AttributeError):
        print("  [warn] Pre-score JSON parse failed — skipping filter", file=sys.stderr)
        return transcript

    if len(scores) != len(segments):
        print(
            f"  [warn] Pre-score returned {len(scores)} scores for {len(segments)} segments "
            "— skipping filter",
            file=sys.stderr,
        )
        return transcript

    kept_pairs = [(seg, scores[i]) for i, seg in enumerate(segments) if float(scores[i]) >= threshold]
    n_dropped = len(segments) - len(kept_pairs)

    if n_dropped > 0:
        kept_dur = sum(s["end"] - s["start"] for s, _ in kept_pairs)
        total_dur = transcript.get("total_duration", sum(s["end"] - s["start"] for s in segments))
        pct = 100 * kept_dur / total_dur if total_dur else 0
        print(
            f"  Filtered {n_dropped}/{len(segments)} low-score segments "
            f"— {len(kept_pairs)} remain ({kept_dur:.0f}s / {total_dur:.0f}s, {pct:.0f}%)",
            file=sys.stderr,
        )
    else:
        print(f"  All {len(segments)} segments passed threshold ≥{threshold:.0f}", file=sys.stderr)

    filtered = dict(transcript)
    filtered["segments"] = [s for s, _ in kept_pairs]
    return filtered


def _process_gpt_selection(
    gpt_result: Dict,
    segments: List[Dict],
    presented_segments: List[Dict],
    shuffle_order: List[int],
    target_duration: float,
) -> Dict:
    """Map GPT selection JSON back to original segment indices and enforce duration window.

    Shared by select_segments() and refine_segments() to avoid duplicating logic.
    """
    cold_open_idx_presented = gpt_result.get("cold_open_index")
    selected_indices_presented = gpt_result.get("selected_indices", [])

    selected_indices_presented = [
        i for i in selected_indices_presented
        if isinstance(i, int) and 0 <= i < len(presented_segments)
    ]

    selected_indices = [shuffle_order[i] for i in selected_indices_presented]
    cold_open_idx = (
        shuffle_order[cold_open_idx_presented]
        if cold_open_idx_presented is not None and 0 <= cold_open_idx_presented < len(presented_segments)
        else None
    )

    if cold_open_idx is not None and cold_open_idx in selected_indices:
        selected_indices = [cold_open_idx] + [i for i in selected_indices if i != cold_open_idx]

    max_dur = target_duration * 1.1
    min_dur = target_duration * 0.9
    total = sum(segments[i]["end"] - segments[i]["start"] for i in selected_indices)
    if total > max_dur:
        trimmed = list(selected_indices)
        while trimmed and sum(segments[i]["end"] - segments[i]["start"] for i in trimmed) > max_dur:
            trimmed.pop()
        selected_indices = trimmed
    elif total < min_dur:
        print(f"  [warn] Selected {total:.0f}s < minimum {min_dur:.0f}s — model under-selected", file=sys.stderr)

    selected_segs = []
    for idx in selected_indices:
        seg = dict(segments[idx])
        seg["_orig_idx"] = idx
        if idx == cold_open_idx:
            seg["_cold_open"] = True
        selected_segs.append(seg)

    return {
        "segments": selected_segs,
        "summary": gpt_result.get("summary", ""),
        "flow_note": gpt_result.get("flow_note", ""),
        "transition_notes": gpt_result.get("transition_notes", []),
        "narrative_map": gpt_result.get("narrative_map", []),
        "sign_off_bleed": gpt_result.get("sign_off_bleed", ""),
        "excluded_reason": gpt_result.get("excluded_reason", ""),
        "total_duration": sum(s["end"] - s["start"] for s in selected_segs),
        "cold_open_index": cold_open_idx,
        "segment_count": len(selected_segs),
    }


def select_segments(
    transcript: Dict,
    topic: str,
    target_duration: float,
    cold_open: bool = True,
    mix: bool = False,
    model: str = "anthropic:claude-sonnet-4-6",
) -> Dict:
    """Use an LLM to select coherent segments from a transcript."""
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
    multi_source = _is_multi_source(segments)
    sources = transcript.get("sources", [])
    source_names = [Path(s).name for s in sources] if sources else []

    # When mix=True, shuffle the presentation order so GPT doesn't have source-file bias.
    # We keep a mapping from shuffled position → original index so returned indices map back.
    if mix and multi_source:
        shuffle_order = list(range(len(segments)))
        random.shuffle(shuffle_order)
        presented_segments = [segments[i] for i in shuffle_order]
    else:
        shuffle_order = list(range(len(segments)))
        presented_segments = segments

    seg_text = _fmt_seg_list(presented_segments)

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

    multi_source_note = ""
    if multi_source:
        names_str = ", ".join(source_names) if source_names else "multiple files"
        if mix:
            multi_source_note = (
                f"\nMULTI-SOURCE NOTE: Segments come from {len(source_names)} different video files "
                f"({names_str}). Each segment is labeled [SRC:filename]. "
                "SEGMENTS HAVE BEEN PRESENTED IN RANDOM ORDER — do NOT assume adjacent segments "
                "in this list are from the same source or in chronological order. "
                "You MUST draw clips from MULTIPLE source files — ideally all of them — "
                "to build the richest possible narrative. Actively prefer alternating between "
                "sources rather than clustering clips from the same file together.\n"
            )
        else:
            multi_source_note = (
                f"\nMULTI-SOURCE NOTE: Segments come from {len(source_names)} different video files "
                f"({names_str}). Each segment is labeled [SRC:filename]. "
                "You may freely interleave clips from different sources to build the best narrative. "
                "Treat each source as a separate speaker or camera angle unless context suggests otherwise.\n"
            )

    user_prompt = f"""Create a {target_duration:.0f}-second highlight reel from this Shopify all-hands transcript.

FOCUS TOPIC: "{topic}"
TARGET DURATION: {target_duration:.0f}s (HARD minimum {target_duration * 0.9:.0f}s, maximum {target_duration * 1.1:.0f}s — you MUST reach the minimum, include good-but-not-perfect clips if needed)
TOTAL SOURCE DURATION: {total_dur:.0f}s{multi_source_note}

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

    print(f"Selecting segments with {model}...", file=sys.stderr)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    raw = response.choices[0].message.content
    try:
        gpt_result = _parse_json_response(raw)
    except json.JSONDecodeError as e:
        return {"error": f"{model} returned invalid JSON: {e}\n{raw[:500]}"}

    result = _process_gpt_selection(
        gpt_result, segments, presented_segments, shuffle_order, target_duration
    )

    # Store conversation state so refine_segments() can continue the multi-turn exchange.
    result["_refine_state"] = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": raw},
        ],
        "segments": segments,
        "presented_segments": presented_segments,
        "shuffle_order": shuffle_order,
        "target_duration": target_duration,
        "model": model,
    }
    return result


def refine_segments(prev_result: Dict, instruction: str, model: Optional[str] = None) -> Dict:
    """Apply a natural-language refinement to a prior selection via multi-turn conversation.

    The model already has the full segment list in its context from select_segments().
    We append the user instruction and ask for a revised selection JSON.

    Args:
        prev_result: Return value from select_segments() or a prior refine_segments() call.
        instruction: Free-text editing direction, e.g. "remove the fundraising tangent".
        model: Override the model (defaults to whatever select_segments() used).

    Returns:
        A new result dict in the same shape as select_segments(), with updated _refine_state.
        Returns prev_result unchanged if the API call fails.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("  [warn] openai not installed", file=sys.stderr)
        return prev_result

    state = prev_result.get("_refine_state")
    if not state:
        print("  [warn] No refinement state — was select_segments() called?", file=sys.stderr)
        return prev_result

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("  [warn] OPENAI_API_KEY not set", file=sys.stderr)
        return prev_result

    base_url = os.environ.get("OPENAI_BASE_URL", "https://proxy.shopify.ai/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)

    model = model or state["model"]
    segments = state["segments"]
    presented_segments = state["presented_segments"]
    shuffle_order = state["shuffle_order"]
    target_duration = state["target_duration"]

    messages = list(state["messages"])
    messages.append({"role": "user", "content": instruction})

    print(f"Refining with {model}...", file=sys.stderr)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    raw = response.choices[0].message.content
    try:
        gpt_result = _parse_json_response(raw)
    except json.JSONDecodeError as e:
        print(f"  [warn] Refinement returned invalid JSON: {e} — keeping prior selection", file=sys.stderr)
        return prev_result

    result = _process_gpt_selection(
        gpt_result, segments, presented_segments, shuffle_order, target_duration
    )

    messages.append({"role": "assistant", "content": raw})
    result["_refine_state"] = {**state, "messages": messages}
    return result


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
        description="AI-powered segment selection for resolve-autocut."
    )
    parser.add_argument("transcript_json", help="Path to transcript JSON from transcribe.py")
    parser.add_argument("--suggest-topics", action="store_true",
                        help="Print N suggested topics from the transcript and exit")
    parser.add_argument("--n-topics", type=int, default=6,
                        help="Number of topics to suggest (default: 6)")
    parser.add_argument("--topic", default=None, help="Focus topic / keywords for selection")
    parser.add_argument("--duration", type=float, required=False, default=None,
                        help="Target duration in seconds (e.g. 180 for 3 minutes)")
    parser.add_argument("--no-cold-open", action="store_true",
                        help="Disable cold open detection")
    parser.add_argument("--mix", action="store_true",
                        help="Shuffle segment presentation to GPT to encourage cross-source mixing")
    parser.add_argument("--model", default="anthropic:claude-sonnet-4-6",
                        help="Model for segment selection (default: anthropic:claude-sonnet-4-6). "
                             "Examples: gpt-4o, gpt-4.1, anthropic:claude-opus-4-6, "
                             "google:gemini-2.5-flash (all via proxy.shopify.ai/v1)")
    parser.add_argument("--no-prescore", action="store_true",
                        help="Skip the pre-scoring pass (use all segments as-is)")
    parser.add_argument("--prescore-model", default="gpt-4.1-mini",
                        help="Cheap model for pre-scoring pass (default: gpt-4.1-mini)")
    parser.add_argument("--prescore-threshold", type=float, default=4.0,
                        help="Minimum score (1–10) to keep a segment (default: 4.0)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="After initial selection, enter a refinement loop (Enter to accept)")
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

    if args.suggest_topics:
        print("Analyzing transcript for topics...", file=sys.stderr)
        topics = suggest_topics(transcript, n=args.n_topics, model=args.model)
        for i, t in enumerate(topics, 1):
            print(f"{i}. {t}")
        sys.exit(0)

    if not args.topic:
        print("Error: --topic is required (or use --suggest-topics)", file=sys.stderr)
        sys.exit(1)

    if not args.duration:
        print("Error: --duration is required", file=sys.stderr)
        sys.exit(1)

    if not args.no_prescore:
        transcript = prescore_segments(
            transcript,
            topic=args.topic,
            threshold=args.prescore_threshold,
            model=args.prescore_model,
        )

    result = select_segments(
        transcript,
        topic=args.topic,
        target_duration=args.duration,
        cold_open=not args.no_cold_open,
        mix=args.mix,
        model=args.model,
    )

    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print_selection_report(result)

    # Iterative refinement REPL
    if args.interactive:
        print("\nRefinement mode — describe changes, or press Enter to accept:", file=sys.stderr)
        while True:
            try:
                instruction = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("", file=sys.stderr)
                break
            if not instruction:
                break
            result = refine_segments(result, instruction, model=args.model)
            if "error" in result:
                print(f"Error: {result['error']}", file=sys.stderr)
                continue
            print_selection_report(result)
            print("\nRefine further, or press Enter to accept:", file=sys.stderr)

    output_data = result["segments"]

    if args.output:
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"Saved {len(output_data)} segments to: {args.output}", file=sys.stderr)
    else:
        json.dump(output_data, sys.stdout, indent=2)
        print()
