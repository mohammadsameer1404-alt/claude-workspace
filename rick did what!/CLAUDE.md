# Rick Dead Watt — YouTube Shorts Pipeline

## Goal
1,000 subscribers + 10M views in 20 days by posting daily high-virality Rick and Morty Shorts.

## Project Structure
```
rick did what!/
  pipeline/
    1_analyze.py         Stage 1 — scene detect + Whisper + audio energy peaks
    2_score.py           Stage 2 — Gemini Flash 3-agent panel scoring (FREE)
    3_export.py          Stage 3 — FFmpeg 9:16 export + pop title + captions
    4_schedule.py        Stage 4 — YouTube API scheduler (5 PM EDT daily)
    scorer_cache.json    Gemini response cache (avoids re-scoring)
    uploaded.json        YouTube upload log
  clips/                 Final exported Shorts go here
  temp/                  Temporary frames (auto-deleted after scoring)
  run_pipeline.py        Master entry point
  requirements.txt       Python dependencies
  .venv/                 Python 3.11 virtual environment
  client_secrets.json    YouTube OAuth credentials (DO NOT share)
```

## Episode Library
- Path: `/Users/sameermohammad/Movies/Rick and Morty`
- Seasons 1–8, 81 episodes total, mixed MP4/MKV
- Season folders: `season 1`–`season 4` (lowercase), `Season 5`–`Season 8` (capitalised)

## Environment Setup
```bash
cd "/Users/sameermohammad/rick did what!"
source .venv/bin/activate
export GEMINI_API_KEY=your-key-here    # free at aistudio.google.com
# client_secrets.json already in project root for YouTube OAuth
```

## Running the Pipeline

```bash
# One episode
python run_pipeline.py --episode "/Users/sameermohammad/Movies/Rick and Morty/season 1/Rick.and.Morty.S01E05.720p.English.Esubs - Vegamovies.to.mp4"

# All 81 episodes in priority order
python run_pipeline.py --all

# Schedule only (clips already exported)
python run_pipeline.py --schedule-only

# Preview schedule without uploading
python run_pipeline.py --schedule-only --dry-run
```

## Priority Episode Order (highest viral density first)
1. S01E05 — Meeseeks and Destroy
2. S01E08 — Rixty Minutes (Interdimensional Cable)
3. S02E04 — Total Rickall
4. S03E03 — Pickle Rick
5. S03E07 — The Ricklantis Mixup (Evil Morty)
6. S02E06 — The Ricks Must Be Crazy
7. S01E01 — Pilot
8. S01E02 — Lawnmower Dog
9. S02E01 — A Rickle in Time
10. S02E10 — The Wedding Squanchers

## 3-Pass Scoring System (2_score.py) — Gemini Flash FREE

| Pass | Resolution | Threshold | Purpose |
|------|-----------|-----------|---------|
| A | 480×270 | ≥ 6.5 | Bulk screen — wide net, nothing good missed |
| B | 720×405 | ≥ 8.0 | Quality gate + generate hook/title/description |
| Panel | 720×405 | ≥ 9.0 | 3-agent debate — all three must agree |

### The 3-Agent Panel
- **Algorithm Expert** — completion rate, hook, replay potential
- **Rick & Morty Fan** — cultural weight, meme potential, fan recognition
- **Devil's Advocate** — argues every reason it will flop
Only clips ALL THREE score ≥ 9.0 are exported.

## Export Quality (3_export.py)
- 1080×1920 (9:16), H.265 M1 hardware, 10 Mbps + 192k AAC
- Pop title: AI hook text overlaid first 2.5s (fades in/out)
- Captions: Whisper transcript burnt-in, white bold + black outline

## Posting Schedule
- **5:00 PM EDT (21:00 UTC)** — peak of your viewer analytics window
- 4 hours before competitors (who post 9–10:45 PM EDT)
- One clip per day via YouTube Data API v3

## API Costs
| Service | Cost |
|---------|------|
| Gemini 1.5 Flash | FREE (1,500 req/day) |
| YouTube Data API | FREE (10,000 units/day) |
| FFmpeg + Whisper | FREE (local, M1 hardware) |

## Key Research
- Seasons 1–3 highest virality; seasons 4–5 lowest — process S1–S3 first
- Best content types: character reactions, one-liners, visual absurdity, Evil Morty arcs
- Optimal Short duration: 30–55 seconds
- Post time derived from your own YouTube Analytics heatmap (GMT+0530)
