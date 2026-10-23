"""Render the ten selected photos with the current semantic Option B settings."""
import json
from pathlib import Path

from PIL import Image

from baseline import process


ROOT = Path(__file__).resolve().parents[1]
rows = json.loads((ROOT / 'review/selection.json').read_text())
destination = ROOT / 'data/review_targets/refined'
destination.mkdir(parents=True, exist_ok=True)
for row in rows:
    with Image.open(ROOT / row['source']) as image:
        result, shadows, contours = process(image, 'semantic', .45, .55, .55)
    for label, output in [('binary', result), ('shadows', shadows), ('contours', contours)]:
        output.save(destination / f"{row['key']}_{label}.png")
    print(row['key'], flush=True)
