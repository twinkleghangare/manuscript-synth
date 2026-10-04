"""Physical deformations applied to the final composited page.

Features:
  - Smooth sinusoidal surface warp (as before, but stronger)
  - Page curl along one edge with perspective-foreshortening and a cast shadow
  - 1–2 fold creases at random angles with bright/dark shading
  - All deformations are applied to the fully-composited image so the
    background and text move together.
  - Safe-area helper: clamp / assert that text bboxes stay within bounds
    after the *warp only* step (curl does not move text boxes).
"""
import cv2
import math
import numpy as np
from PIL import Image


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_float(img):
    return np.array(img, dtype=np.float32)


def _from_float(arr, mode="RGB"):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode)


# ── surface warp ─────────────────────────────────────────────────────────────

def surface_warp(img, rng, amp=6.0):
    """Low-frequency sinusoidal warp — keeps text inside the canvas."""
    a = np.array(img)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    p1, p2 = rng.uniform(0, 6.28, 2)
    freq_y = float(rng.uniform(80, 180))
    freq_x = float(rng.uniform(120, 260))
    dx = amp * np.sin(yy / freq_y + p1)
    dy = amp * np.sin(xx / freq_x + p2)
    # secondary harmonic for slight ripple
    dx += (amp * 0.3) * np.sin(yy / (freq_y * 0.4) + p2)
    dy += (amp * 0.3) * np.sin(xx / (freq_x * 0.4) + p1)
    src_x = np.clip(xx + dx, 0, w - 1).astype(np.float32)
    src_y = np.clip(yy + dy, 0, h - 1).astype(np.float32)
    out = cv2.remap(a, src_x, src_y, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(out)


# ── fold creases ─────────────────────────────────────────────────────────────

def _fold_crease(arr, rng):
    """Add 1–2 fold creases across the image."""
    h, w = arr.shape[:2]
    n_folds = int(rng.integers(1, 3))  # 1 or 2 folds

    for _ in range(n_folds):
        # random line through the page
        angle = float(rng.uniform(-20, 20))   # degrees from horizontal
        cy    = float(rng.uniform(h * 0.20, h * 0.80))
        cx    = float(rng.uniform(w * 0.10, w * 0.90))

        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # signed distance from the fold line
        dist = (xx - cx) * (-sin_a) + (yy - cy) * cos_a

        crease_w = float(rng.uniform(4, 14))
        bright_side = float(rng.uniform(1.04, 1.12))
        dark_side   = float(rng.uniform(0.80, 0.92))

        # bright near-line highlight
        highlight = np.exp(-0.5 * (dist / crease_w) ** 2)
        # darker shadow on one side
        shadow = np.where(dist > 0,
                          1.0 - 0.15 * np.clip(1 - dist / (crease_w * 3), 0, 1),
                          1.0)

        factor = (1.0 + (bright_side - 1.0) * highlight) * shadow
        arr = np.clip(arr * factor[..., None], 0, 255)

    return arr


# ── page curl ────────────────────────────────────────────────────────────────

def _page_curl(img, rng):
    """Curl one corner of the page with perspective foreshortening + shadow."""
    a = _to_float(img)
    h, w = a.shape[:2]

    # choose which corner to curl
    corner = int(rng.integers(4))   # 0=TL 1=TR 2=BL 3=BR
    curl_frac_x = float(rng.uniform(0.06, 0.18))
    curl_frac_y = float(rng.uniform(0.06, 0.18))
    cx = int(curl_frac_x * w)
    cy = int(curl_frac_y * h)

    # coordinates depend on which corner
    if corner == 0:    # top-left
        fold_x, fold_y = cx, cy
        xs, xe, ys, ye = 0, fold_x, 0, fold_y
        flip_x, flip_y = 1, 1
    elif corner == 1:  # top-right
        fold_x, fold_y = w - cx, cy
        xs, xe, ys, ye = fold_x, w, 0, fold_y
        flip_x, flip_y = -1, 1
    elif corner == 2:  # bottom-left
        fold_x, fold_y = cx, h - cy
        xs, xe, ys, ye = 0, fold_x, fold_y, h
        flip_x, flip_y = 1, -1
    else:              # bottom-right
        fold_x, fold_y = w - cx, h - cy
        xs, xe, ys, ye = fold_x, w, fold_y, h
        flip_x, flip_y = -1, -1

    # darken the curled region (simulates the page bending away)
    for row in range(ys, ye):
        for col in range(xs, xe):
            dist_x = abs(col - fold_x) / max(cx, 1)
            dist_y = abs(row - fold_y) / max(cy, 1)
            t = min(dist_x, dist_y)
            # foreshortening: compress slightly
            dark = 1.0 - 0.45 * t ** 1.5
            a[row, col, :] = np.clip(a[row, col, :] * dark, 0, 255)

    # cast shadow just inside the fold
    shadow_w = max(4, int(min(cx, cy) * 0.35))
    if corner == 0:
        rx, ry = range(fold_x, min(w, fold_x + shadow_w)), range(fold_y, min(h, fold_y + shadow_w))
    elif corner == 1:
        rx, ry = range(max(0, fold_x - shadow_w), fold_x), range(fold_y, min(h, fold_y + shadow_w))
    elif corner == 2:
        rx, ry = range(fold_x, min(w, fold_x + shadow_w)), range(max(0, fold_y - shadow_w), fold_y)
    else:
        rx, ry = range(max(0, fold_x - shadow_w), fold_x), range(max(0, fold_y - shadow_w), fold_y)

    for row in ry:
        for col in rx:
            t = min(abs(col - fold_x) / shadow_w, abs(row - fold_y) / shadow_w)
            a[row, col, :] = np.clip(a[row, col, :] * (0.65 + 0.35 * t), 0, 255)

    return _from_float(a, img.mode)


# ── safe-area guard ───────────────────────────────────────────────────────────

def clamp_bbox(box, size, margin=8):
    """Clamp a (x0, y0, x1, y1) box so it stays within (w, h) minus margin."""
    w, h = size
    x0, y0, x1, y1 = box
    x0 = max(margin, min(x0, w - margin))
    y0 = max(margin, min(y0, h - margin))
    x1 = max(x0 + 1, min(x1, w - margin))
    y1 = max(y0 + 1, min(y1, h - margin))
    return (x0, y0, x1, y1)


def assert_bbox_inside(box, size, label="box"):
    """Raise AssertionError if any part of box is outside the canvas."""
    w, h = size
    x0, y0, x1, y1 = box
    assert x0 >= 0 and y0 >= 0 and x1 <= w and y1 <= h, (
        f"bbox overflow: {label} {box} outside canvas {size}"
    )


# ── public API ────────────────────────────────────────────────────────────────

def warp(img, rng, amp=6.0, do_curl=True, do_folds=True):
    """Apply surface warp, optional fold creases, and optional page curl.

    The warp is applied first (so text bboxes can be validated before curl),
    then folds, then curl.

    Args:
        img:      PIL Image (RGB or RGBA)
        rng:      numpy Generator
        amp:      sinusoidal warp amplitude in pixels
        do_curl:  whether to apply page-curl effect
        do_folds: whether to add fold creases
    Returns:
        PIL Image in same mode as input
    """
    # 1. Surface warp (background + text together)
    img = surface_warp(img, rng, amp=amp)

    # 2. Fold creases
    if do_folds and rng.random() < 0.70:
        arr = np.array(img, dtype=np.float32)
        arr = _fold_crease(arr, rng)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), img.mode)

    # 3. Page curl (only 50 % of pages)
    if do_curl and rng.random() < 0.50:
        img = _page_curl(img, rng)

    return img
