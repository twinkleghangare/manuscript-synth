"""Ink colour, bleed, smudges, faded strokes, and occasional blots.

Design:
  - ink_color()       returns a randomised ink RGB tuple (dark brown / iron-gall,
                      carbon black, or red/vermilion for highlights)
  - apply_artifacts() operates on an RGBA text layer:
      1. Bleed: Gaussian blur composited back to spread ink slightly
      2. Stroke-weight jitter via morphological erosion/dilation
      3. Patchy fading via low-freq noise on alpha channel
      4. Smudges: dark semi-transparent blobs smeared in a direction
      5. Occasional ink blot: round dark splotch
      6. Dried ink sheen: slight brightening along stroke centres
"""
import cv2
import numpy as np
from PIL import Image, ImageFilter

# ── palette ───────────────────────────────────────────────────────────────────
# Each tuple: (R, G, B)  — all dark; variations added per-call
INKS = [
    (22,  18,  14),   # deep carbon black
    (40,  28,  18),   # iron-gall brown-black
    (58,  42,  28),   # warm brown iron-gall
    (32,  32,  46),   # cool blue-black
    (50,  35,  22),   # medium brown
    (28,  22,  16),   # neutral dark
]
HIGHLIGHT_INKS = [
    (148, 28, 22),    # vermilion / red ochre
    (160, 40, 20),    # cinnabar
    (130, 20, 15),    # deep red
]


def ink_color(rng, highlight=False):
    """Return a randomised ink RGB tuple."""
    palette = HIGHLIGHT_INKS if highlight else INKS
    base = np.array(palette[int(rng.integers(len(palette)))], dtype=np.float32)
    jitter = rng.integers(-14, 15, 3).astype(np.float32)
    return tuple(int(c) for c in np.clip(base + jitter, 0, 255))


# ── artifact helpers ──────────────────────────────────────────────────────────

def _bleed(layer_arr, rng):
    """Gaussian ink bleed: blend a slightly blurred copy back in."""
    radius = float(rng.uniform(0.8, 2.0))
    blurred = cv2.GaussianBlur(layer_arr, (0, 0), radius)
    # blend 30–55 % toward blurred
    alpha = float(rng.uniform(0.30, 0.55))
    out = layer_arr.astype(np.float32) * (1 - alpha) + blurred.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _stroke_weight_jitter(layer_arr, rng):
    """Randomly erode or dilate strokes to simulate pressure variation."""
    choice = rng.random()
    k_size = int(rng.choice([3, 5]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    alpha = layer_arr[..., 3]
    if choice < 0.4:    # slightly thinner strokes
        alpha = cv2.erode(alpha, kernel, iterations=1)
    elif choice < 0.7:  # slightly thicker
        alpha = cv2.dilate(alpha, kernel, iterations=1)
    # else: no change
    out = layer_arr.copy()
    out[..., 3] = alpha
    return out


def _patchy_fading(layer_arr, rng):
    """Low-frequency noise fades parts of the ink (uneven preservation)."""
    h, w = layer_arr.shape[:2]
    # two-scale noise for broad patches + fine grained
    coarse_h, coarse_w = max(2, h // 20), max(2, w // 20)
    fine_h,   fine_w   = max(2, h // 6),  max(2, w // 6)

    coarse = rng.random((coarse_h, coarse_w)).astype(np.float32)
    coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)

    fine   = rng.random((fine_h, fine_w)).astype(np.float32)
    fine   = cv2.resize(fine, (w, h), interpolation=cv2.INTER_CUBIC)

    noise  = 0.65 * coarse + 0.35 * fine
    # map noise to [min_opacity, 1.0]
    min_op = float(rng.uniform(0.55, 0.85))
    opacity = min_op + (1.0 - min_op) * noise

    out = layer_arr.copy().astype(np.float32)
    out[..., 3] = np.clip(out[..., 3] * opacity, 0, 255)
    return out.astype(np.uint8)


def _smudges(layer_arr, rng, max_smudges=3):
    """Directional smear: motion-blur a small region of the alpha channel."""
    h, w = layer_arr.shape[:2]
    n_smudges = int(rng.integers(0, max_smudges + 1))
    out = layer_arr.copy()
    for _ in range(n_smudges):
        x0 = int(rng.integers(0, w))
        y0 = int(rng.integers(0, h))
        sw = int(rng.integers(30, 120))
        sh = int(rng.integers(8, 35))
        x1, y1 = min(w, x0 + sw), min(h, y0 + sh)
        region = out[y0:y1, x0:x1, :]
        if region.size == 0:
            continue
        angle = float(rng.uniform(-15, 15))
        klen  = int(rng.integers(7, 21))
        # build motion kernel
        kern = np.zeros((klen, klen), np.float32)
        kern[klen // 2, :] = 1.0 / klen
        M = cv2.getRotationMatrix2D((klen // 2, klen // 2), angle, 1)
        kern = cv2.warpAffine(kern, M, (klen, klen))
        kern /= kern.sum() + 1e-9
        # apply to alpha only, partially
        smear_strength = float(rng.uniform(0.25, 0.55))
        orig_alpha = region[..., 3].astype(np.float32)
        smeared    = cv2.filter2D(orig_alpha, -1, kern)
        out[y0:y1, x0:x1, 3] = np.clip(
            orig_alpha * (1 - smear_strength) + smeared * smear_strength, 0, 255
        ).astype(np.uint8)
    return out


def _ink_blots(img_rgba, rng, max_blots=2):
    """Add occasional round ink blots directly onto the RGBA layer."""
    n_blots = int(rng.integers(0, max_blots + 1))
    arr = np.array(img_rgba, dtype=np.uint8)
    h, w = arr.shape[:2]
    for _ in range(n_blots):
        cx = int(rng.integers(w // 10, w * 9 // 10))
        cy = int(rng.integers(h // 10, h * 9 // 10))
        r  = int(rng.integers(5, 20))
        halo_r = r + int(rng.integers(3, 8))
        ry = int(max(r * rng.uniform(0.6, 1.2), 1))
        halo_ry = int(max(halo_r * rng.uniform(0.6, 1.2), 1))

        x0, x1 = max(0, cx - halo_r - 2), min(w, cx + halo_r + 3)
        y0, y1 = max(0, cy - halo_ry - 2), min(h, cy + halo_ry + 3)
        if x1 <= x0 or y1 <= y0:
            continue

        color_rgb = ink_color(rng, highlight=False)
        alpha_val = int(rng.integers(170, 240))

        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist_sq = ((xx - cx) / float(r)) ** 2 + ((yy - cy) / float(ry)) ** 2
        halo_dist_sq = ((xx - cx) / float(halo_r)) ** 2 + ((yy - cy) / float(halo_ry)) ** 2

        mask = dist_sq <= 1.0
        halo = (halo_dist_sq <= 1.0) & ~mask

        patch = arr[y0:y1, x0:x1]
        patch[mask, 0] = color_rgb[0]
        patch[mask, 1] = color_rgb[1]
        patch[mask, 2] = color_rgb[2]
        patch[mask, 3] = alpha_val

        patch[halo, 0] = np.clip(color_rgb[0] + 30, 0, 255)
        patch[halo, 1] = np.clip(color_rgb[1] + 20, 0, 255)
        patch[halo, 2] = np.clip(color_rgb[2] + 15, 0, 255)
        patch[halo, 3] = np.clip(patch[halo, 3].astype(int) + 50, 0, 255).astype(np.uint8)

        arr[y0:y1, x0:x1] = patch

    return Image.fromarray(arr, "RGBA")


# ── public API ────────────────────────────────────────────────────────────────

def apply_artifacts(layer, rng, cfg=None):
    """Apply all ink aging artifacts to an RGBA text layer.

    Pipeline:
      bleed → stroke-weight jitter → patchy fading → smudges → blots
    """
    cfg = cfg or {}
    arr = np.array(layer)

    if cfg.get("bleed", True):
        arr = _bleed(arr, rng)
    arr = _stroke_weight_jitter(arr, rng)
    if cfg.get("fading", True):
        arr = _patchy_fading(arr, rng)
    if cfg.get("smudges", True):
        smudges_cfg = cfg.get("smudges")
        max_smudges = int(smudges_cfg.get("count_range", [1, 3])[1]) if isinstance(smudges_cfg, dict) else int(cfg.get("max_smudges", 3))
        arr = _smudges(arr, rng, max_smudges=max_smudges)

    result = Image.fromarray(arr, "RGBA")
    if cfg.get("blots", True):
        blots_cfg = cfg.get("blots")
        max_blots = int(blots_cfg.get("count_range", [1, 2])[1]) if isinstance(blots_cfg, dict) else int(cfg.get("max_blots", 2))
        result = _ink_blots(result, rng, max_blots=max_blots)
    return result
