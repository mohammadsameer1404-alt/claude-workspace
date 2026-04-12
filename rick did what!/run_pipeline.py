"""
Rick Dead Watt — YouTube Shorts Pipeline Orchestrator

Full pipeline for one episode:
  Stage 1: Analyze     — scene detect + Whisper + audio peaks → candidates_raw.json
  Stage 2: Score       — Claude Haiku 480p bulk filter → candidates.json  (low credits)
  Stage 2b: Review     — Claude Sonnet 720p final pass on winners          (quality gate)
  Stage 3: Export      — FFmpeg premium 9:16 export + styled captions → clips/
  Stage 4: Schedule    — YouTube API daily scheduling at 5 PM EST

Usage:
  # Process one episode (full pipeline)
  python run_pipeline.py --episode episodes/S01E05.mp4

  # Process ALL episodes in episodes/ folder (priority order)
  python run_pipeline.py --all

  # Schedule already-exported clips (no re-processing)
  python run_pipeline.py --schedule-only

  # Dry-run: preview schedule without uploading
  python run_pipeline.py --schedule-only --dry-run
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

BASE_DIR     = Path(__file__).parent
EPISODES_DIR = Path("/Users/sameermohammad/Movies/Rick and Morty")  # your existing library
PIPELINE_DIR = BASE_DIR / "pipeline"
CLIPS_DIR    = BASE_DIR / "clips"
TEMP_DIR     = BASE_DIR / "temp"
UPLOAD_LOG   = PIPELINE_DIR / "uploaded.json"

# Season subfolders inside EPISODES_DIR (mixed capitalisation as on disk)
SEASON_DIRS = [
    "season 1", "season 2", "season 3", "season 4",
    "Season 5", "Season 6", "Season 7", "Season 8",
]

# Research-backed priority order — highest viral density first
# SxxExx pattern works regardless of dots vs spaces in filename
PRIORITY_STEMS = [
    "S01E05", "S01E08", "S02E04", "S03E03",
    "S03E07", "S02E06", "S01E01", "S01E02",
    "S02E01", "S02E10",
]

PYTHON = sys.executable   # same interpreter that launched this script


def extract_code(path: Path) -> str:
    """Extract normalised SxxExx from any filename variant (dots or spaces)."""
    import re
    m = re.search(r"S(\d{2})E(\d{2})", path.stem, re.IGNORECASE)
    return f"S{m.group(1).upper()}E{m.group(2).upper()}" if m else path.stem


def run_stage(script: str, *args) -> bool:
    """Run a pipeline stage script. Returns True on success."""
    cmd = [PYTHON, str(PIPELINE_DIR / script)] + list(args)
    print(f"\n  $ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def get_raw_json(code: str) -> Path:
    return PIPELINE_DIR / f"{code}_candidates_raw.json"


def get_candidates_json(code: str) -> Path:
    return PIPELINE_DIR / f"{code}_candidates.json"


def all_episodes() -> list[Path]:
    """Walk every season subfolder and return all video files."""
    exts = {".mp4", ".mkv", ".avi", ".mov"}
    files = []
    for season in SEASON_DIRS:
        season_path = EPISODES_DIR / season
        if season_path.exists():
            files.extend(
                f for f in sorted(season_path.iterdir())
                if f.suffix.lower() in exts
            )
    return files


def sorted_episodes() -> list[Path]:
    """Return all episode files: priority episodes first, then rest in season/ep order."""
    files = all_episodes()
    code_map = {extract_code(f): f for f in files}

    priority = [code_map[s] for s in PRIORITY_STEMS if s in code_map]
    priority_set = set(priority)
    rest = [f for f in files if f not in priority_set]
    return priority + rest


def process_episode(ep: Path) -> bool:
    code = extract_code(ep)
    print(f"\n{'═'*60}")
    print(f"  EPISODE : {ep.name}")
    print(f"  CODE    : {code}")
    print(f"{'═'*60}")

    # Stage 1: narrative-arc sliding windows + word timestamps
    ok = run_stage("1_analyze.py", "--episode", str(ep), "--output", str(PIPELINE_DIR))
    if not ok or not get_raw_json(code).exists():
        print(f"  [FAIL] Stage 1 failed for {ep.name}")
        return False

    # Stage 2: 3-frame scoring, story-arc prompts, hard cap 3 clips
    ok = run_stage("2_score.py",
                   "--raw",    str(get_raw_json(code)),
                   "--output", str(PIPELINE_DIR),
                   "--temp",   str(TEMP_DIR))
    if not ok or not get_candidates_json(code).exists():
        print(f"  [FAIL] Stage 2 failed for {ep.name}")
        return False

    # Stage 3: export with neon green captions + word-pop
    ok = run_stage("3_export.py",
                   "--candidates", str(get_candidates_json(code)),
                   "--output",     str(CLIPS_DIR))
    if not ok:
        print(f"  [WARN] Stage 3 had errors for {ep.name} — check clips/")

    # Stage 4: 3-AI virality audit (Gemini + Groq + Mistral, all free)
    run_stage("4_audit.py",
              "--clips",    str(CLIPS_DIR),
              "--pipeline", str(PIPELINE_DIR))

    print(f"\n  Clips ready → {CLIPS_DIR}/")
    print(f"  Review them and upload manually when satisfied.")
    return True


def schedule(dry_run: bool = False) -> None:
    args = ["--clips", str(CLIPS_DIR), "--log", str(UPLOAD_LOG)]
    if dry_run:
        args.append("--dry-run")
    run_stage("4_schedule.py", *args)


def print_status() -> None:
    ep_count   = len(all_episodes())
    clip_count = len(list(CLIPS_DIR.glob("*.mp4")))
    scheduled  = 0
    if UPLOAD_LOG.exists():
        scheduled = len(json.loads(UPLOAD_LOG.read_text()))

    print("\n── Current Status ──────────────────────────────────────────")
    print(f"  Episodes found (all seasons) : {ep_count}")
    print(f"  Clips exported  in clips/    : {clip_count}")
    print(f"  Scheduled on YouTube         : {scheduled}")
    print(f"\n  Library path: {EPISODES_DIR}")
    print("\nExamples:")
    ep = all_episodes()[0] if all_episodes() else Path("S01E05.mp4")
    print(f'  python run_pipeline.py --episode "{ep}"')
    print("  python run_pipeline.py --all")
    print("  python run_pipeline.py --schedule-only --dry-run")


def main():
    for d in [PIPELINE_DIR, CLIPS_DIR, TEMP_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Rick Dead Watt — YouTube Shorts Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--episode",       help="Path to episode file to process")
    g.add_argument("--all",           action="store_true", help="Process all episodes/ in priority order")
    g.add_argument("--schedule-only", action="store_true", help="Only schedule clips/, skip processing")
    parser.add_argument("--dry-run",  action="store_true", help="Preview schedule without uploading")
    args = parser.parse_args()

    if args.schedule_only:
        schedule(dry_run=args.dry_run)

    elif args.episode:
        ep = Path(args.episode)
        if not ep.exists():
            print(f"File not found: {args.episode}")
            sys.exit(1)
        process_episode(ep)

    elif args.all:
        episodes = sorted_episodes()
        if not episodes:
            print(f"No episode files found in {EPISODES_DIR}/")
            print("Drop your MP4/MKV episode files there and rerun.")
            sys.exit(1)
        print(f"\n{len(episodes)} episodes queued (priority order):")
        for ep in episodes:
            print(f"  {ep.name}")
        for ep in episodes:
            process_episode(ep)
        print("\nAll done. Review clips/ and upload manually when satisfied.")

    else:
        parser.print_help()
        print_status()


if __name__ == "__main__":
    main()
