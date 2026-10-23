"""Resumable batch export of semantic Option B from prepared normalized images."""
import argparse
from pathlib import Path

from PIL import Image

from baseline import contact_sheet, process


def save_png(image, path):
    temporary = path.with_name(path.stem + '.tmp.png')
    image.save(temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/normalized'))
    parser.add_argument('--output', type=Path, default=Path('data/option_b'))
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--reverse', action='store_true', help='Process filenames in reverse order')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--ink', type=float, default=0.45)
    parser.add_argument('--contour', type=float, default=0.55)
    parser.add_argument('--detail', type=float, default=0.55)
    args = parser.parse_args()
    paths = sorted(args.input.glob('*.png'), reverse=args.reverse)
    if args.limit is not None:
        paths = paths[:args.limit]
    for index, path in enumerate(paths, 1):
        binary_path = args.output / f'{path.stem}.png'
        shadow_path = Path('data/masks/semantic_shadows') / f'{path.stem}.png'
        contour_path = Path('data/masks/semantic_contours') / f'{path.stem}.png'
        comparison_path = args.output / 'comparisons' / f'{path.stem}.png'
        outputs = (binary_path, shadow_path, contour_path, comparison_path)
        if not args.force and all(p.exists() for p in outputs):
            print(f'{index}/{len(paths)} existing {path.stem}', flush=True)
            continue
        for output in outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as image:
            binary, shadows, contours = process(image, 'semantic', args.ink, args.contour, args.detail)
            save_png(binary, binary_path)
            save_png(shadows, shadow_path)
            save_png(contours, contour_path)
            save_png(contact_sheet(image, {'option b': binary}, cell_width=480), comparison_path)
        print(f'{index}/{len(paths)} rendered {path.stem}', flush=True)


if __name__ == '__main__':
    main()
