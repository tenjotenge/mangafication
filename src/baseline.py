"""CPU-only, strictly binary photograph-to-ink baselines."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

METHODS = ('global', 'adaptive', 'color', 'composite')
SEMANTIC_METHOD = 'semantic'


def _clean(mask, min_area=12):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    valid = stats[:, cv2.CC_STAT_AREA] >= min_area
    valid[0] = False
    return valid[labels].astype(np.uint8) * 255


def process(image, method='composite', ink=0.45, contour=0.55, detail=0.55):
    """Return binary output, shadow mask, contour mask; 0 is ink, 255 is paper."""
    if method == SEMANTIC_METHOD:
        try:
            from .semantic import render
        except ImportError:  # direct script invocation
            from semantic import render
        return render(image, ink, contour, detail)
    if method not in METHODS:
        raise ValueError(f'Unknown method: {method}')
    ink, contour, detail = [float(np.clip(v, 0, 1)) for v in (ink, contour, detail)]
    rgb = np.asarray(image.convert('RGB'))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    luminance = lab[:, :, 0]
    smooth = cv2.bilateralFilter(luminance, 7, 55, 55)
    threshold = int(np.percentile(smooth, 15 + 70 * ink))
    if method == 'global':
        shadow = (smooth <= threshold).astype(np.uint8) * 255
    elif method == 'adaptive':
        local = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 41, int(13 - 17 * ink))
        shadow = _clean(local)
    else:
        # Lab chroma distance retains boundaries between equally bright colors.
        chroma = lab[:, :, 1:3].astype(np.float32)
        chroma_smooth = cv2.GaussianBlur(chroma, (7, 7), 0)
        delta = np.linalg.norm(chroma - chroma_smooth, axis=2)
        local = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 51, int(18 - 15 * ink))
        chromatic = (delta > 7 + 10 * (1 - detail)).astype(np.uint8) * 255
        if method == 'color':
            shadow = _clean(cv2.bitwise_or(local, chromatic))
        else:
            shadow = _clean(cv2.morphologyEx(local, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)))
    if method == 'color':
        # Chroma gradient explicitly contributes even when luminance is unchanged.
        ab = cv2.GaussianBlur(lab[:, :, 1:3], (5, 5), 0).astype(np.float32)
        gx = cv2.Sobel(ab, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(ab, cv2.CV_32F, 0, 1)
        chroma_edges = np.sqrt((gx * gx + gy * gy).sum(axis=2))
    else:
        chroma_edges = np.zeros(luminance.shape, np.float32)
    light_edges = cv2.Canny(smooth, int(45 + 90 * (1 - detail)), int(100 + 150 * (1 - detail)))
    chroma_cutoff = np.percentile(chroma_edges, 85 + 13 * (1 - detail)) if chroma_edges.any() else float('inf')
    edges = cv2.bitwise_or(light_edges, (chroma_edges > chroma_cutoff).astype(np.uint8) * 255)
    width = 1 + int(contour > 0.66) + int(contour > 0.9)
    edges = cv2.dilate(edges, np.ones((width, width), np.uint8))
    if contour < 0.1:
        edges[:] = 0
    if method == 'global':
        edges[:] = 0
    if method == 'adaptive':
        edges[:] = 0
    if method == 'composite':
        # Dark masses plus independently controlled linework.
        combined = cv2.bitwise_or(shadow, edges)
    elif method == 'color':
        combined = cv2.bitwise_or(shadow, edges)
    else:
        combined = shadow
    to_paper = lambda ink_mask: np.where(ink_mask > 0, 0, 255).astype(np.uint8)
    return Image.fromarray(to_paper(combined)), Image.fromarray(to_paper(shadow)), Image.fromarray(to_paper(edges))


def contact_sheet(image, variants, cell_width=360):
    original = image.convert('RGB')
    entries = [('Original', original)] + [(name.title(), value.convert('RGB')) for name, value in variants.items()]
    height = max(round(im.height * cell_width / im.width) for _, im in entries)
    sheet = Image.new('RGB', (cell_width * len(entries), height + 32), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, (label, im) in enumerate(entries):
        im.thumbnail((cell_width, height), Image.Resampling.LANCZOS)
        sheet.paste(im, (index * cell_width + (cell_width - im.width)//2, 32))
        draw.text((index * cell_width + 8, 9), label, fill='black')
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--output', type=Path, default=Path('data/comparisons'))
    parser.add_argument('--semantic', action='store_true', help='Include optional pretrained semantic variant')
    parser.add_argument('--ink', type=float, default=0.45)
    parser.add_argument('--contour', type=float, default=0.55)
    parser.add_argument('--detail', type=float, default=0.55)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with Image.open(args.image) as image:
        variants = {}
        for method in METHODS + ((SEMANTIC_METHOD,) if args.semantic else ()):
            result, shadows, contours = process(image, method, args.ink, args.contour, args.detail)
            variants[method] = result
            result.save(args.output / f'{args.image.stem}_{method}.png')
            if method in ('composite', SEMANTIC_METHOD):
                shadows.save(args.output / f'{args.image.stem}_{method}_shadows.png')
                contours.save(args.output / f'{args.image.stem}_{method}_contours.png')
        contact_sheet(image, variants).save(args.output / f'{args.image.stem}_comparison.png')


if __name__ == '__main__':
    main()
