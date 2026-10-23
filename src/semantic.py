"""Optional CPU semantic guidance for geometry-preserving two-tone rendering."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parents[1] / 'data/models/onnx/model.onnx'


@lru_cache(maxsize=1)
def _session():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError('Install onnxruntime in the project .venv for semantic mode') from exc
    if not MODEL_PATH.exists():
        raise RuntimeError('Semantic model missing. Run: python src/download_model.py')
    return ort.InferenceSession(str(MODEL_PATH), providers=['CPUExecutionProvider'])


def clothing_labels(image):
    """Return 18-class human parsing labels at the source image resolution."""
    rgb = image.convert('RGB')
    array = np.asarray(rgb.resize((512, 512), Image.Resampling.BILINEAR), np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    values = np.transpose((array - mean) / std, (2, 0, 1))[None].astype(np.float32)
    logits = _session().run(None, {'pixel_values': values})[0][0]
    # Interpolate logits before argmax to avoid blocky 128-pixel label edges.
    best_score = np.full((rgb.height, rgb.width), -np.inf, np.float32)
    labels = np.zeros((rgb.height, rgb.width), np.uint8)
    for class_id, channel in enumerate(logits):
        score = cv2.resize(channel, rgb.size, interpolation=cv2.INTER_LINEAR)
        better = score > best_score
        labels[better] = class_id
        best_score[better] = score[better]
    return labels


def render(image, ink=0.45, contour=0.55, detail=0.55):
    """Return binary rendering and separately exportable shadow and contour masks."""
    rgb = np.asarray(image.convert('RGB'))
    labels = clothing_labels(image)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lum = lab[:, :, 0]
    smooth = cv2.bilateralFilter(lum, 7, 55, 55)
    subject = labels != 0
    shadow = np.zeros(lum.shape, np.uint8)
    # Classify the stage from its border, where the foreground model occupies
    # relatively little area. This avoids mistaking a bright outfit for scenery.
    bg = ~subject
    if bg.any():
        height, width = smooth.shape
        border = np.concatenate((smooth[:max(1, height//8)].ravel(),
                                 smooth[:, :max(1, width//10)].ravel(),
                                 smooth[:, -max(1, width//10):].ravel()))
        if np.median(border) < 90:
            # A flat black stage is easier to draw from than photographic
            # pools of light or irregular white holes around dark clothing.
            shadow[bg] = 255
        # Light stages remain white; their edge lines are added below.
    for class_id in np.unique(labels):
        if class_id == 0:
            continue
        region = labels == class_id
        if region.sum() < 20:
            continue
        values = smooth[region]
        median = np.median(values)
        if class_id == 2:  # Keep blonde and midtone hair highlights as paper.
            coverage = (10 + 30 * ink) if median > 130 else ((35 + 20 * ink) if median > 55 else (82 + 12 * ink))
            cutoff = np.percentile(values, coverage)
        elif class_id in (11, 12, 13, 14, 15):  # Keep skin as white negative space.
            cutoff = np.percentile(values, 8 + 22 * ink)
        elif median < 35:  # Nearly black cloth: solid mass with narrow highlights.
            cutoff = np.percentile(values, 88 + 10 * ink)
        elif median < 95:  # Feather masses: reserve more white highlights.
            cutoff = np.percentile(values, 70 + 15 * ink)
        elif median > 175:  # Light fabric: select only dark folds.
            cutoff = np.percentile(values, 7 + 28 * ink)
        else:
            cutoff = np.percentile(values, 30 + 48 * ink)
        shadow[region & (smooth < cutoff)] = 255
    # Remove tiny patches without erasing small jewellery or facial marks.
    count, components, stats, _ = cv2.connectedComponentsWithStats(shadow, 8)
    valid = stats[:, cv2.CC_STAT_AREA] >= max(4, int((1-detail) * 16))
    valid[0] = False
    shadow = valid[components].astype(np.uint8) * 255
    edges = cv2.Canny(smooth, int(45 + 85*(1-detail)), int(105 + 130*(1-detail)))
    # A parsing boundary restores pose and fabric silhouette despite close colors.
    silhouette = cv2.morphologyEx(subject.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) * 255
    edges = cv2.bitwise_or(edges, silhouette)
    width = 1 + int(contour > 0.66) + int(contour > 0.9)
    if contour < 0.1:
        edges[:] = 0
    elif width > 1:
        edges = cv2.dilate(edges, np.ones((width, width), np.uint8))
    combined = cv2.bitwise_or(shadow, edges)
    # Thin white strokes reveal low-contrast folds inside black fabric. CLAHE
    # acts only as an edge detector; the exported result is still strictly binary.
    enhanced = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(smooth)
    fold_edges = cv2.Canny(enhanced, int(60 + 40*(1-detail)), int(125 + 60*(1-detail)))
    fabric = subject & ~np.isin(labels, (2, 11, 12, 13, 14, 15)) & (smooth < 105)
    relief = (fold_edges > 0) & fabric & (shadow > 0)
    count, components, stats, _ = cv2.connectedComponentsWithStats(relief.astype(np.uint8), 8)
    keep = stats[:, cv2.CC_STAT_AREA] >= 12
    keep[0] = False
    combined[keep[components]] = 0
    # White relief separates dark clothing from an equally dark background.
    inner_border = subject & ~cv2.erode(subject.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    dark_outside = cv2.dilate((bg & (shadow > 0)).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    combined[inner_border & dark_outside & (shadow > 0)] = 0
    paper = lambda mask: Image.fromarray(np.where(mask > 0, 0, 255).astype(np.uint8))
    return paper(combined), paper(shadow), paper(edges)
