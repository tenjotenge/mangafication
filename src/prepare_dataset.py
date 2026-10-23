"""Normalize downloaded images, export masks, and make duplicate-safe splits."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from baseline import process, contact_sheet, METHODS


def dhash(image):
    gray = ImageOps.grayscale(image).resize((17, 16), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray)
    return sum(int(bit) << index for index, bit in enumerate((pixels[:, 1:] > pixels[:, :-1]).flatten()))


def components(items, max_distance=8):
    parent = list(range(len(items)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(items)):
        for j in range(i):
            if items[i]['sha256'] == items[j]['sha256'] or (items[i]['dhash'] ^ items[j]['dhash']).bit_count() <= max_distance:
                parent[root(i)] = root(j)
    groups = {}
    for i in range(len(items)):
        groups.setdefault(root(i), []).append(items[i])
    return list(groups.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('data/raw/manifest.jsonl'))
    parser.add_argument('--max-side', type=int, default=1536)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-fraction', type=float, default=0.18)
    parser.add_argument('--val-fraction', type=float, default=0.12)
    parser.add_argument('--near-duplicate-distance', type=int, default=8)
    args = parser.parse_args()
    if args.max_side < 64 or args.test_fraction + args.val_fraction >= 1:
        parser.error('Invalid size or split fractions')
    records = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    good = [r for r in records if r['status'] in ('downloaded', 'existing', 'duplicate_url') and Path(r['path']).exists()]
    unique = {r['path']: r for r in good}
    locations = [Path('data/normalized'), Path('data/masks/shadows'), Path('data/masks/contours'), Path('data/comparisons'), Path('data/splits')]
    for directory in locations:
        directory.mkdir(parents=True, exist_ok=True)
    items = []
    for path, record in unique.items():
        with Image.open(path) as source:
            original = ImageOps.exif_transpose(source).convert('RGB')
            image = original.copy()
            image.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
            name = Path(path).stem
            normalized = Path('data/normalized') / f'{name}.png'
            image.save(normalized)
            output, shadows, contours = process(image)
            shadows.save(Path('data/masks/shadows') / f'{name}.png')
            contours.save(Path('data/masks/contours') / f'{name}.png')
            variants = {method: process(image, method)[0] for method in METHODS}
            contact_sheet(image, variants).save(Path('data/comparisons') / f'{name}.png')
            items.append({'id': name, 'raw': path, 'normalized': str(normalized),
                          'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                          'dhash': dhash(original), 'metadata': record['metadata']})
    groups = components(items, args.near_duplicate_distance)
    random.Random(args.seed).shuffle(groups)
    total = len(items)
    targets = {'test': round(total * args.test_fraction), 'val': round(total * args.val_fraction)}
    splits = {'train': [], 'val': [], 'test': []}
    for group in groups:
        if len(splits['test']) < targets['test']:
            destination = 'test'
        elif len(splits['val']) < targets['val']:
            destination = 'val'
        else:
            destination = 'train'
        splits[destination].extend(group)
    for name, entries in splits.items():
        path = Path('data/splits') / f'{name}.jsonl'
        with path.open('w') as handle:
            for entry in entries:
                serialized = {k: v for k, v in entry.items() if k != 'dhash'}
                handle.write(json.dumps(serialized) + '\n')
        print(f'{name}: {len(entries)} -> {path}')
    print(f'{len(groups)} near-duplicate groups across {total} images')


if __name__ == '__main__':
    main()
