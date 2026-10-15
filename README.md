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
puckbot-analyze src/puckbot/vision/data/samples/sample_hockey_table_top_view.png
```

```
  table   : found, playing surface 42.5 x 82.5 in
  puck    : pixel (555, 846) -> x=23.5 in, y=49.3 in, ~3.6 in across
  mallet 1: pixel (292, 1418) -> x=6.2 in, y=80.5 in, ~4.3 in across
  all good
```

If something is missing, it says what and how to fix it.

| Option | Saves |
| --- | --- |
| `--out result.png` | the photo with everything drawn on it + a top-down view |
| `--steps steps.png` | one picture showing every pipeline step, in order |

Positions: origin = surface corner at the top-left of the photo; x across the table, y along it.

## Code

Vision code (finding the puck, mallets and table) is in `src/puckbot/vision/`.
Start with its [README](src/puckbot/vision/README.md).
