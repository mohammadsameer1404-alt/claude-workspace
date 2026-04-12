"""
Stage 4: YouTube API Scheduler

For each clip in clips/:
  1. Opens the clip in QuickTime Player so you can review it
  2. Asks: upload this? (y / skip / quit)
  3. Shows a suggested publish time and asks for date + time (UTC)
  4. Asks for final confirmation before uploading
  5. Uploads and logs to uploaded.json — skips clips already there

Setup (one-time):
  1. Go to Google Cloud Console → APIs & Services → Credentials
  2. Create OAuth 2.0 Client ID (Desktop app type)
  3. Download as client_secrets.json and place it in this project root
  4. First run will open a browser for consent — token saved to token.json

Usage:
  python pipeline/4_schedule.py --clips clips/ --dry-run   # preview
  python pipeline/4_schedule.py --clips clips/             # live upload
"""

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# YouTube Data API scope for uploads
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

# Target: 5:00 PM EDT = 21:00 UTC (summer) / 22:00 UTC (winter EST)
# Analytics shows peak viewer activity 2:30–8:30 PM EDT.
# 5 PM EDT is the sweet spot: peak window + 4 hrs ahead of competitors (who post 9–10:45 PM EDT).
# EDT = UTC-4 (Mar–Nov), EST = UTC-5 (Nov–Mar)
PUBLISH_HOUR_UTC   = 21    # 5 PM EDT (summer) — adjust to 22 in winter (EST)
PUBLISH_MINUTE_UTC = 0     # on the hour — clean scheduling

DEFAULT_DESCRIPTION = (
    "Rick and Morty | #Shorts\n\n"
    "#RickAndMorty #RickDeadWatt #AdultSwim #Shorts #rickmorty #funny #animated"
)
DEFAULT_TAGS = [
    "rick and morty",
    "rick and morty shorts",
    "rick morty funny",
    "adult swim",
    "animation shorts",
    "rick morty clips",
    "shorts",
    "RickDeadWatt",
]


def authenticate(secrets_file: str = "client_secrets.json", token_file: str = "token.json") -> object:
    """Return an authenticated YouTube API service object."""
    creds = None

    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(secrets_file):
                raise FileNotFoundError(
                    f"OAuth secrets file not found: {secrets_file}\n"
                    "Download your OAuth 2.0 client credentials from Google Cloud Console\n"
                    "and save as client_secrets.json in the project root."
                )
            flow = InstalledAppFlow.from_client_secrets_file(secrets_file, SCOPES)
            creds = flow.run_local_server(port=0)
            with open(token_file, "w") as f:
                f.write(creds.to_json())
            print(f"  Auth token saved to {token_file}")

    return build("youtube", "v3", credentials=creds)


def load_uploaded(log_path: str) -> dict:
    if os.path.exists(log_path):
        with open(log_path) as f:
            return json.load(f)
    return {}


def save_uploaded(log_path: str, log: dict) -> None:
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)


def next_publish_slot(uploaded_log: dict) -> datetime:
    """
    Return the next available publish datetime at PUBLISH_HOUR_UTC.
    Starts from tomorrow (never today — API requires 15+ min in future and we want a full day).
    """
    now = datetime.now(timezone.utc)
    candidate = now.replace(hour=PUBLISH_HOUR_UTC, minute=PUBLISH_MINUTE_UTC, second=0, microsecond=0)

    # Start from tomorrow at minimum
    candidate += timedelta(days=1)

    # Advance past any already-scheduled slots
    scheduled_dates = set()
    for entry in uploaded_log.values():
        if "publish_at" in entry:
            dt = datetime.fromisoformat(entry["publish_at"])
            scheduled_dates.add(dt.date())

    while candidate.date() in scheduled_dates:
        candidate += timedelta(days=1)

    return candidate


def load_clip_metadata(clip_path: str, pipeline_dir: str) -> dict:
    """
    Look up the AI-generated title, description, and hook for this clip
    from the candidates JSON produced by stage 2.
    Falls back to generic values if not found.
    """
    clip_id   = Path(clip_path).stem           # e.g. S01E05_120_178
    parts     = clip_id.split("_")
    episode   = parts[0] if parts else "S01E01"
    candidates_path = os.path.join(pipeline_dir, f"{episode}_candidates.json")

    if os.path.exists(candidates_path):
        with open(candidates_path) as f:
            data = json.load(f)
        for clip in data.get("clips", []):
            if clip.get("clip_id") == clip_id:
                return {
                    "title":       clip.get("youtube_title", f"Rick and Morty 💀 #Shorts"),
                    "description": clip.get("description",  "Rick and Morty being unhinged 😭\n\n#RickAndMorty #Shorts #RickDeadWatt #AdultSwim #rickmorty #animation #funny #viral"),
                }

    return {
        "title":       f"Rick and Morty — {episode} 💀 #Shorts",
        "description": "Rick and Morty being unhinged 😭\n\n#RickAndMorty #Shorts #RickDeadWatt #AdultSwim #rickmorty #animation #funny #viral",
    }


def upload_clip(
    youtube,
    clip_path: str,
    publish_at: datetime,
    pipeline_dir: str,
    dry_run: bool = False,
) -> dict:
    """Upload a single clip scheduled to publish at publish_at. Returns upload result."""

    meta        = load_clip_metadata(clip_path, pipeline_dir)
    title       = meta["title"]
    description = meta["description"]
    publish_iso = publish_at.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    print(f"  {'[DRY RUN] ' if dry_run else ''}Scheduling: {Path(clip_path).name}")
    print(f"    Title      : {title}")
    print(f"    Publish at : {publish_iso}  (5:00 PM EDT / peak viewer window)")
    print(f"    File size  : {os.path.getsize(clip_path) / 1_000_000:.1f} MB")

    if dry_run:
        return {"id": "DRY_RUN", "publish_at": publish_iso, "title": title}

    body = {
        "snippet": {
            "title":       title,
            "description": description,
            "tags":        DEFAULT_TAGS,
            "categoryId":  "24",   # Entertainment
        },
        "status": {
            "privacyStatus":           "private",
            "publishAt":               publish_iso,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(clip_path, mimetype="video/mp4", resumable=True, chunksize=5 * 1024 * 1024)

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            print(f"    Uploading... {pct}%", end="\r")

    print(f"    Uploaded ✓  video ID: {response['id']}")
    return {"id": response["id"], "publish_at": publish_iso, "title": title}


def preview_clip(clip_path: str) -> None:
    """Open the clip in QuickTime Player for review, then wait for the user to close it."""
    import subprocess
    print(f"\n  Opening preview: {Path(clip_path).name}")
    print("  (QuickTime Player will open — watch the clip, then come back here)")
    subprocess.Popen(["open", "-W", "-a", "QuickTime Player", clip_path])


def ask_publish_datetime(suggested: datetime) -> datetime:
    """
    Interactively ask for publish date and time.
    Shows a suggested default; accepts blank input to use it.
    Returns a timezone-aware UTC datetime.
    """
    suggested_local = suggested.strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n  Suggested publish time: {suggested_local}")
    print("  (Press Enter to accept, or type your own)")

    while True:
        date_str = input("  Publish date (YYYY-MM-DD) [Enter = suggested]: ").strip()
        time_str = input("  Publish time (HH:MM, 24-hr UTC)  [Enter = suggested]: ").strip()

        if not date_str and not time_str:
            return suggested

        try:
            use_date = datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else suggested.date()
            if time_str:
                h, m = map(int, time_str.split(":"))
            else:
                h, m = suggested.hour, suggested.minute
            result = datetime(use_date.year, use_date.month, use_date.day, h, m, 0, tzinfo=timezone.utc)
            # Must be at least 15 minutes in the future
            if result <= datetime.now(timezone.utc) + timedelta(minutes=15):
                print("  ✗ That time is too soon — YouTube requires at least 15 minutes in the future.")
                continue
            return result
        except ValueError:
            print("  ✗ Invalid format. Use YYYY-MM-DD for date and HH:MM for time.")


def run(clips_dir: str, log_path: str, dry_run: bool = False) -> None:
    pipeline_dir = os.path.dirname(log_path)
    clips = sorted(Path(clips_dir).glob("*.mp4"))
    if not clips:
        print(f"No .mp4 files found in {clips_dir}")
        return

    uploaded_log = load_uploaded(log_path)

    pending = [c for c in clips if str(c) not in uploaded_log]
    already = len(clips) - len(pending)

    print(f"\n[4_schedule] Clips found      : {len(clips)}")
    print(f"[4_schedule] Already scheduled: {already}")
    print(f"[4_schedule] To schedule      : {len(pending)}")

    if not pending:
        print("  Nothing new to schedule.")
        return

    if not dry_run:
        youtube = authenticate()
    else:
        youtube = None
        print("  [DRY RUN MODE — no actual uploads]")

    for clip_path in pending:
        print(f"\n{'─'*55}")
        print(f"  Clip: {clip_path.name}")
        meta = load_clip_metadata(str(clip_path), pipeline_dir)
        print(f"  Title      : {meta['title']}")
        size_mb = clip_path.stat().st_size / 1_000_000
        print(f"  File size  : {size_mb:.1f} MB")

        # ── Step 1: preview ──────────────────────────────────────────────────
        preview_clip(str(clip_path))

        # ── Step 2: confirm or skip ──────────────────────────────────────────
        print()
        choice = input("  Upload this clip? [y = yes / s = skip / q = quit]: ").strip().lower()
        if choice == "q":
            print("  Stopping. Clips scheduled so far have been saved.")
            break
        if choice != "y":
            print("  Skipped.")
            continue

        # ── Step 3: ask for publish date + time ──────────────────────────────
        suggested = next_publish_slot(uploaded_log)
        publish_at = ask_publish_datetime(suggested)
        publish_iso = publish_at.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        print(f"\n  Will publish at: {publish_iso}")
        confirm = input("  Confirm upload? [y/n]: ").strip().lower()
        if confirm != "y":
            print("  Cancelled — moving to next clip.")
            continue

        # ── Step 4: upload ────────────────────────────────────────────────────
        try:
            result = upload_clip(youtube, str(clip_path), publish_at, pipeline_dir, dry_run=dry_run)
            uploaded_log[str(clip_path)] = {
                **result,
                "clip_file": clip_path.name,
                "scheduled_on": datetime.now(timezone.utc).isoformat(),
            }
            if not dry_run:
                save_uploaded(log_path, uploaded_log)
        except Exception as e:
            print(f"  [error] Failed to schedule {clip_path.name}: {e}")

    save_uploaded(log_path, uploaded_log)
    print(f"\n  Schedule complete. Log saved to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: Schedule clips to YouTube at 5 PM EST daily")
    parser.add_argument("--clips",   default="clips",         help="Directory containing final .mp4 clips")
    parser.add_argument("--log",     default="pipeline/uploaded.json", help="Upload log JSON path")
    parser.add_argument("--dry-run", action="store_true",     help="Preview schedule without uploading")
    args = parser.parse_args()

    run(args.clips, args.log, dry_run=args.dry_run)
