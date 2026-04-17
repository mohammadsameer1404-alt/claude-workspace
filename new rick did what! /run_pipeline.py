#!/usr/bin/env python3
"""
Rick Tedwatt pipeline — edit + audit in one command.

Stage 1 (research) is done by the rick-and-morty-fan skill in Claude.
This script handles Stage 2 (edit) + Stage 3 (audit).

Usage:
    python3 run_pipeline.py <script.json> [--hook "Hook Title Here"] [--threshold 8.6]

Output:
    output/passed/   — clips that passed all 3 AIs at the threshold
    output/rejected/ — clips moved here if you choose at the prompt
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
EDITOR      = Path.home() / ".claude/skills/rick-and-morty-shorts/scripts/editor.py"
AUDITOR     = Path.home() / ".claude/skills/shorts-auditor/scripts/audit.py"
SKILL_DIR   = Path.home() / ".claude/skills/rick-and-morty-shorts"
OUTPUT_DIR  = PIPELINE_DIR / "output"
PASSED_DIR  = OUTPUT_DIR / "passed"
RAW_DIR     = OUTPUT_DIR / "raw"


def main() -> int:
    parser = argparse.ArgumentParser(description="Rick Tedwatt edit + audit pipeline")
    parser.add_argument("script", help="Path to clip spec JSON from the research skill")
    parser.add_argument("--hook", default=None, help="Hook title (auto-generated if omitted)")
    parser.add_argument("--threshold", type=float, default=8.0, help="Virality gate threshold (default 8.0)")
    parser.add_argument("--audit-only", action="store_true", help="Skip editing, audit existing clips in output/raw/")
    args = parser.parse_args()

    script = Path(args.script).expanduser().resolve()
    if not args.audit_only and not script.exists():
        print(f"ERROR: script not found: {script}"); return 1

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PASSED_DIR.mkdir(parents=True, exist_ok=True)

    if not args.audit_only:
        # ── Stage 2: Edit ──────────────────────────────────────────────
        print(f"\n{'─'*50}")
        print(f"STAGE 2 — EDIT")
        print(f"{'─'*50}")

        spec = json.loads(script.read_text())
        reel_id = spec.get("reel_id", script.stem)

        hook = args.hook
        if not hook:
            # auto-generate a simple fallback hook from the spec
            hook = spec.get("hook_title") or f"Rick and Morty — {reel_id.replace('_', ' ').title()}"

        print(f"  Script : {script}")
        print(f"  Hook   : {hook}")
        print(f"  Output : {RAW_DIR}/")
        print()

        r = subprocess.run(
            [sys.executable, str(EDITOR), str(script), str(RAW_DIR), str(SKILL_DIR), hook]
        )
        if r.returncode != 0:
            print("\nSTAGE 2 FAILED — fix the errors above then re-run."); return r.returncode
        print("\nStage 2 complete.")

    # ── Stage 3: Audit ────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"STAGE 3 — AUDIT  (threshold {args.threshold})")
    print(f"{'─'*50}\n")

    raw_clips = list(RAW_DIR.glob("*.mp4")) + list(RAW_DIR.glob("*.mov"))
    if not raw_clips:
        print(f"No clips found in {RAW_DIR}"); return 0

    print(f"  Clips to audit: {len(raw_clips)}")
    print(f"  Passed folder : {PASSED_DIR}/\n")

    r = subprocess.run(
        [sys.executable, str(AUDITOR), str(RAW_DIR), str(PASSED_DIR), str(args.threshold)]
    )
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
