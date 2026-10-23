"""Make binary synthetic references and four-column review sheets for ten photos."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / 'review' / 'selection.json'
GENERATED = ROOT / 'data' / 'review_targets' / 'generated'
SHEETS = ROOT / 'data' / 'review_targets' / 'sheets'


def binary_copy(source: Path, destination: Path) -> float:
    with Image.open(source) as image:
        gray = np.asarray(image.convert('L'))
    # Fixed threshold is reproducible and keeps small ink details in the raw art.
    pixels = np.where(gray < 160, 0, 255).astype(np.uint8)
    Image.fromarray(pixels, 'L').save(destination)
    return float(np.mean(pixels == 0))


def fitted(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert('RGB')
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (width, height), 'white')
    canvas.paste(image, ((width-image.width)//2, (height-image.height)//2))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell-width', type=int, default=400)
    parser.add_argument('--cell-height', type=int, default=600)
    args = parser.parse_args()
    SHEETS.mkdir(parents=True, exist_ok=True)
    rows = json.loads(SELECTION.read_text())
    metrics = []
    columns = 5
    overview = Image.new('RGB', (args.cell_width * columns, (args.cell_height + 32) * len(rows)), 'white')
    for row_index, row in enumerate(rows):
        key = row['key']
        paths = [ROOT / row['source'], ROOT / row['baseline'], ROOT / 'data/review_targets/refined' / f'{key}_binary.png']
        for variant in ('A', 'B'):
            raw = GENERATED / f'{key}_{variant}_raw.png'
            binary = GENERATED / f'{key}_{variant}.png'
            coverage = binary_copy(raw, binary)
            paths.append(binary)
            metrics.append({'key': key, 'variant': variant, 'black_coverage': round(coverage, 4)})
        sheet = Image.new('RGB', (args.cell_width * columns, args.cell_height + 32), 'white')
        draw = ImageDraw.Draw(sheet)
        for column, (label, path) in enumerate(zip(('Original', 'Previous Option B', 'Refined Option B', 'Generated A', 'Generated B'), paths)):
            x = column * args.cell_width
            draw.text((x+8, 8), f'{key}  {label}', fill='black')
            sheet.paste(fitted(path, args.cell_width, args.cell_height), (x, 32))
        sheet.save(SHEETS / f'{key}.jpg', quality=90)
        overview.paste(sheet, (0, row_index*(args.cell_height+32)))
    overview.save(SHEETS / 'all_ten.jpg', quality=85)
    (SHEETS / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(f'Built {len(rows)} review sheets and {len(metrics)} strictly binary synthetic images')


if __name__ == '__main__':
    main()
