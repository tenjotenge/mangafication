"""Resumable downloader for the selected runway photographs."""
import argparse
import csv
import hashlib
import io
import json
import random
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError


def read_selection(path):
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        candidates = [name for name in fields if name.lower() in ('url', 'image_url', 'photo_url', 'image')]
        if not candidates:
            raise ValueError(f'No image URL column in {fields}')
        rows = list(reader)
    return rows, candidates[0]


def filename(row, url):
    key = ''.join(c for c in row.get('key', '') if c.isalnum())[:24] or 'image'
    digest = hashlib.sha256(url.encode()).hexdigest()[:12]
    return f'{key}_{digest}.jpg'


def validate(payload):
    with Image.open(io.BytesIO(payload)) as image:
        image.verify()
    with Image.open(io.BytesIO(payload)) as image:
        if image.format not in ('JPEG', 'PNG', 'WEBP'):
            raise ValueError(f'Unexpected image format: {image.format}')
        image.load()
        return image.width, image.height, image.format


def fetch(url, timeout, attempts):
    parsed = urlparse(url)
    if parsed.scheme != 'https' or not parsed.netloc:
        raise ValueError('Only HTTPS image URLs are accepted')
    for attempt in range(attempts):
        try:
            req = Request(url, headers={'User-Agent': 'mangafication-research/0.1', 'Accept': 'image/*'})
            with urlopen(req, timeout=timeout) as response:
                content_type = response.headers.get_content_type()
                if not content_type.startswith('image/'):
                    raise ValueError(f'Unexpected HTTP content type: {content_type}')
                payload = response.read(30_000_001)
                if len(payload) > 30_000_000:
                    raise ValueError('Image exceeds 30 MB safety limit')
            return payload, validate(payload)
        except HTTPError as exc:
            if exc.code in (401, 403, 404, 410):
                raise
            error = exc
        except (URLError, TimeoutError, OSError) as exc:
            error = exc
        if attempt + 1 < attempts:
            time.sleep(min(20, 2 ** attempt) + random.uniform(0, 0.5))
    raise error


def write_manifest(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=Path('training_selection.csv'))
    parser.add_argument('--output', type=Path, default=Path('data/raw/mugler'))
    parser.add_argument('--manifest', type=Path, default=Path('data/raw/manifest.jsonl'))
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--timeout', type=float, default=20)
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--delay', type=float, default=0.5, help='Minimum delay between requests')
    args = parser.parse_args()
    rows, url_column = read_selection(args.selection)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    seen = {}
    for row in rows[:args.limit]:
        url = row[url_column].strip()
        record = {'metadata': row, 'url': url, 'path': '', 'status': '', 'error': ''}
        if url in seen:
            record.update(status='duplicate_url', path=seen[url])
        else:
            path = args.output / filename(row, url)
            record['path'] = str(path)
            try:
                if path.exists():
                    dimensions = validate(path.read_bytes())
                    record['status'] = 'existing'
                else:
                    payload, dimensions = fetch(url, args.timeout, args.attempts)
                    temporary = path.with_suffix('.part')
                    temporary.write_bytes(payload)
                    temporary.replace(path)
                    record['status'] = 'downloaded'
                    time.sleep(args.delay)
                record['width'], record['height'], record['format'] = dimensions
                record['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                seen[url] = str(path)
            except (HTTPError, URLError, TimeoutError, OSError, ValueError, UnidentifiedImageError) as exc:
                record.update(status='failed', error=str(exc))
                path.with_suffix('.part').unlink(missing_ok=True)
        records.append(record)
        write_manifest(args.manifest, records)
        print(f"{len(records)}/{min(len(rows), args.limit or len(rows))} {record['status']} {record['path']} {record['error']}", flush=True)
    print(f'Manifest: {args.manifest}')


if __name__ == '__main__':
    main()
