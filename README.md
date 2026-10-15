# Puckbot

An autonomous air hockey player for a full-size table: overhead camera → laptop (vision + policy)
→ ESP32 → stepper drivers → CoreXY gantry. See `CLAUDE.md` for the full design and plan.

## Setup

```bash
git clone https://github.com/uscmakers/puckbot.git
cd puckbot
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Analyse a photo of the table

```bash
puckbot-analyze data/samples/sample_hockey_table_top_view.png --out outputs/
```

```
  table   : found, playing surface 42.5 x 82.5 in
  puck    : pixel (555, 846) -> x=23.5 in, y=49.3 in, ~3.6 in across
  mallet 1: pixel (292, 1420) -> x=6.2 in, y=80.6 in, ~4.9 in across
  all good
```

It finds the blue playing surface (→ pixel-to-inch mapping), the puck and the mallets on the table,
and prints positions in inches. If something is missing it says what and how to fix it. `--out`
saves the photo with everything drawn on it next to a straightened top-down view.

Origin = the playing-surface corner at the top-left of the photo; x runs across the table, y along it.
All settings (table size, colours, puck/mallet sizes in pixels) are in `configs/our_table.json`.

## Layout

```
src/puckbot/vision/
  analyze_image.py   the script: photo -> table, puck, mallets (puckbot-analyze)
  puck_detector.py   finds round blobs of a colour (used for both puck and mallets)
  calibration.py     pixels <-> table mm from the 4 corners of the blue surface
configs/our_table.json
data/samples/        reference photo (tracked); everything else in data/ is git-ignored
tests/
```
