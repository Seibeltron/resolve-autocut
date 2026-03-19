#!/usr/bin/env python3
"""
utils.py — Shared utilities for resolve-autocut scripts.
"""

import json


def parse_json_response(raw: str) -> dict:
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
