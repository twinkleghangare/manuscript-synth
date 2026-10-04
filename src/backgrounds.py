"""Procedural backgrounds: aged handmade paper and palm leaf manuscript.

Paper features:
  - Irregular fibrous texture (multi-octave noise)
  - Uneven yellowing / aging gradient
  - Tea-stain / water-damage blobs
  - Foxing spots (brown mold dots)
  - Torn or frayed edges
  - Faint ruled guide lines

Palm leaf features:
  - Horizontal vein striations
  - String holes (two, centered vertically)
  - Cracked / darkened edges
  - 2-3 tone variants

All parameters are randomised per call so no two outputs look alike.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ── base tone palettes ────────────────────────────────────────────────────────
PAPER_TONES = [
    (222, 200, 160),
    (210, 184, 140),
    (232, 214, 178),
    (198, 170, 126),
    (240, 220, 180),
    (215, 195, 155),
]
PALM_TONES = [
    (176, 140,  80),
    (160, 122,  66),
    (188, 152,  96),
    (200, 165, 100),
    (152, 118,  58),
]


# ── low-level helpers ─────────────────────────────────────────────────────────

def _fractal_noise(rng, h, w, octaves=6, persistence=0.5):
    """Layered smooth noise → [0,1] float32 array."""
    out, amp, tot = np.zeros((h, w), np.float32), 1.0, 0.0
    for o in range(octaves):
        s_h = max(2, h >> (octaves - o - 1))
        s_w = max(2, w >> (octaves - o - 1))
        small = rng.random((s_h, s_w)).astype(np.float32)
        layer = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        out += amp * layer
        tot += amp
        amp *= persistence
    return out / tot


def _smooth_blob(rng, h, w, cx, cy, rx, ry, intensity=1.0):
    """Return a float32 map with a soft elliptical blob at (cx, cy)."""
    yy, xx = np.ogrid[:h, :w]
    d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    blob = np.clip(1.0 - d, 0, 1).astype(np.float32) ** 2
    return blob * intensity


def _apply_vignette(img_f, h, w, inner=0.35, outer=0.10):
    """Edge-darkening vignette; inner / outer control the ramp width."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = np.minimum(xx, w - xx) / w
    dy = np.minimum(yy, h - yy) / h
    d  = np.minimum(dx, dy)
    mask = np.clip(d / inner, 0, 1)
    return img_f * (outer + (1.0 - outer) * mask)[..., None]


# ── paper module ──────────────────────────────────────────────────────────────

def _make_fiber_texture(rng, h, w):
    """Simulate handmade paper fibers via anisotropic noise."""
    # horizontal fibers: stretch noise strongly in X
    fiber_h = max(2, h // 4)
    fiber_w = max(2, w // 2)
    raw = rng.random((fiber_h, fiber_w)).astype(np.float32)
    raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
    raw = cv2.GaussianBlur(raw, (1, 21), 0)   # elongate horizontally

    # cross fibers
    raw2 = rng.random((fiber_h * 2, fiber_w // 2)).astype(np.float32)
    raw2 = cv2.resize(raw2, (w, h), interpolation=cv2.INTER_LINEAR)
    raw2 = cv2.GaussianBlur(raw2, (21, 1), 0)  # elongate vertically

    return (0.7 * raw + 0.3 * raw2)


def _add_tea_stains(rng, img_f, h, w):
    """Large water-damage / tea-wash blobs."""
    n_stains = int(rng.integers(2, 6))
    for _ in range(n_stains):
        cx = float(rng.uniform(0, w))
        cy = float(rng.uniform(0, h))
        rx = float(rng.uniform(w * 0.05, w * 0.25))
        ry = float(rng.uniform(h * 0.05, h * 0.20))
        blob = _smooth_blob(rng, h, w, cx, cy, rx, ry, intensity=float(rng.uniform(0.08, 0.25)))
        # stain is a warm brownish darkening
        darkening = np.stack([blob * rng.uniform(0.85, 0.98),
                               blob * rng.uniform(0.80, 0.93),
                               blob * rng.uniform(0.60, 0.80)], axis=-1)
        img_f = img_f * (1.0 - darkening) + img_f * darkening * 0.7
    return img_f


def _add_foxing(rng, img_f, h, w):
    """Small brownish mold/age spots scattered across the page."""
    n_spots = int(rng.integers(20, 60))
    for _ in range(n_spots):
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))
        rx = float(rng.uniform(3, 16))
        ry = float(rx * rng.uniform(0.6, 1.4))
        intensity = float(rng.uniform(0.3, 0.85))

        x0, x1 = max(0, int(cx - rx * 1.5)), min(w, int(cx + rx * 1.5 + 1))
        y0, y1 = max(0, int(cy - ry * 1.5)), min(h, int(cy + ry * 1.5 + 1))
        if x1 <= x0 or y1 <= y0:
            continue

        yy, xx = np.ogrid[y0:y1, x0:x1]
        d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        blob = np.clip(1.0 - d, 0, 1).astype(np.float32) ** 2 * intensity

        patch = img_f[y0:y1, x0:x1]
        fr = patch[..., 0] * (1.0 - blob * 0.35) + blob * 120.0 / 255.0
        fg = patch[..., 1] * (1.0 - blob * 0.45) + blob * 80.0 / 255.0
        fb = patch[..., 2] * (1.0 - blob * 0.55) + blob * 40.0 / 255.0
        img_f[y0:y1, x0:x1] = np.clip(np.stack([fr, fg, fb], axis=-1), 0, 1)
    return img_f


def _add_ruled_lines(draw, h, w, rng, margin=90):
    """Faint horizontal guide lines a scribe would have used."""
    n_lines = int(rng.integers(8, 20))
    spacing = (h - 2 * margin) // (n_lines + 1)
    alpha = int(rng.integers(18, 45))
    color = (
        int(rng.integers(100, 160)),
        int(rng.integers(80, 130)),
        int(rng.integers(50, 90)),
        alpha,
    )
    for i in range(1, n_lines + 1):
        y = margin + i * spacing + int(rng.integers(-4, 5))
        x0 = margin + int(rng.integers(-10, 10))
        x1 = w - margin + int(rng.integers(-10, 10))
        draw.line([(x0, y), (x1, y)], fill=color, width=1)


def _torn_edge_mask(rng, h, w, roughness=0.012, border=18):
    """Return a float32 alpha mask that fades out near the page border with a torn look."""
    noise_w = rng.random(w).astype(np.float32)
    noise_w = cv2.GaussianBlur(noise_w.reshape(1, -1), (1, 21), 0).reshape(-1)
    fringe_w = (border + noise_w * border * 1.5).astype(np.float32)

    noise_h = rng.random(h).astype(np.float32)
    noise_h = cv2.GaussianBlur(noise_h.reshape(1, -1), (1, 21), 0).reshape(-1)
    fringe_h = (border + noise_h * border * 1.5).astype(np.float32)

    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]

    # Top & bottom fade
    top_mask = np.clip((yy - fringe_w[None, :]) / border, 0, 1)
    bot_mask = np.clip(((h - 1 - yy) - fringe_w[None, :]) / border, 0, 1)

    # Left & right fade
    left_mask  = np.clip((xx - fringe_h[:, None]) / border, 0, 1)
    right_mask = np.clip(((w - 1 - xx) - fringe_h[:, None]) / border, 0, 1)

    return np.minimum.reduce([top_mask, bot_mask, left_mask, right_mask])


def _make_paper(rng, w, h, cfg=None):
    """Full aged handmade paper background."""
    base_tone = np.array(PAPER_TONES[int(rng.integers(len(PAPER_TONES)))], np.float32) / 255.0

    # base fractal texture
    n = _fractal_noise(rng, h, w, octaves=7)
    img_f = base_tone[None, None, :] * (0.72 + 0.50 * n[..., None])

    # fiber overlay
    fiber = _make_fiber_texture(rng, h, w)
    img_f *= (0.88 + 0.22 * fiber[..., None])

    # uneven yellowing gradient
    yy = np.linspace(0, 1, h).reshape(h, 1)
    yellowing = rng.uniform(0.90, 1.0) - yy * rng.uniform(0.0, 0.12)
    img_f[..., 0] *= (yellowing + rng.uniform(0.0, 0.04))
    img_f[..., 1] *= (yellowing * 0.95)
    img_f[..., 2] *= (yellowing * 0.82)

    # tea / age stains
    img_f = _add_tea_stains(rng, img_f, h, w)

    # foxing spots
    img_f = _add_foxing(rng, img_f, h, w)

    # edge vignette / aging
    img_f = _apply_vignette(img_f, h, w, inner=0.20, outer=0.55)
    img_f = np.clip(img_f, 0, 1)

    # convert to PIL RGBA
    img_u8 = (img_f * 255).astype(np.uint8)
    pil = Image.fromarray(img_u8, "RGB").convert("RGBA")

    # ruled guide lines (drawn on RGBA overlay)
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    _add_ruled_lines(ImageDraw.Draw(overlay), h, w, rng)
    pil = Image.alpha_composite(pil, overlay)

    # torn / frayed edges (set alpha near border)
    if rng.random() < 0.75:  # 75 % of sheets have frayed edges
        mask = _torn_edge_mask(rng, h, w, border=int(rng.integers(12, 28)))
        arr = np.array(pil)
        arr[..., 3] = (arr[..., 3].astype(np.float32) * mask).astype(np.uint8)
        pil = Image.fromarray(arr, "RGBA")

    return pil


# ── palm leaf module ──────────────────────────────────────────────────────────

def _make_palm_leaf(rng, w, h, cfg=None):
    """Realistic palm leaf folio with veins, string holes, cracked edges."""
    base_tone = np.array(PALM_TONES[int(rng.integers(len(PALM_TONES)))], np.float32) / 255.0

    # base gradient: palm leaves are slightly darker at top and bottom
    n = _fractal_noise(rng, h, w, octaves=5)
    img_f = base_tone[None, None, :] * (0.70 + 0.45 * n[..., None])

    # ── horizontal vein striations ──────────────────────────────────────────
    # thin horizontal bands of lighter/darker color
    n_veins = int(rng.integers(30, 60))
    vein_positions = sorted(rng.uniform(0, h, n_veins).tolist())
    for yv in vein_positions:
        width = rng.uniform(1, 4)
        bright = rng.uniform(-0.08, 0.10)
        y0 = int(max(0, yv - width))
        y1 = int(min(h, yv + width + 1))
        img_f[y0:y1, :, :] += bright

    # secondary micro-vein noise (very horizontal)
    mv_raw = rng.random((h, max(2, w // 40))).astype(np.float32)
    mv = cv2.resize(mv_raw, (w, h), interpolation=cv2.INTER_LINEAR)
    mv = cv2.GaussianBlur(mv, (1, 71), 0)
    img_f *= (0.88 + 0.20 * mv[..., None])

    # ── two or three tone depth shading ────────────────────────────────────
    n_tones = int(rng.integers(2, 4))
    for _ in range(n_tones):
        cx = float(rng.uniform(0, w))
        cy = float(rng.uniform(0, h))
        blob = _smooth_blob(rng, h, w, cx, cy,
                            float(rng.uniform(w * 0.1, w * 0.35)),
                            float(rng.uniform(h * 0.1, h * 0.35)),
                            float(rng.uniform(0.05, 0.15)))
        img_f *= (1 - blob[..., None] * rng.uniform(0.05, 0.20))

    # ── edge cracks and darkening ───────────────────────────────────────────
    # strong vignette to darken edges
    img_f = _apply_vignette(img_f, h, w, inner=0.14, outer=0.35)

    # crack lines along top/bottom edges
    crack_color = float(-0.25)
    for edge_y in [int(h * 0.04), int(h * 0.96)]:
        n_cracks = int(rng.integers(3, 9))
        for _ in range(n_cracks):
            start_x = int(rng.uniform(0, w))
            length  = int(rng.uniform(w * 0.02, w * 0.15))
            for px in range(start_x, min(w, start_x + length)):
                drift = edge_y + int(rng.integers(-3, 4))
                y_lo = max(0, drift - 2)
                y_hi = min(h, drift + 3)
                img_f[y_lo:y_hi, px, :] = np.clip(img_f[y_lo:y_hi, px, :] + crack_color, 0, 1)

    img_f = np.clip(img_f, 0, 1)

    # ── string holes ────────────────────────────────────────────────────────
    img_u8 = (img_f * 255).astype(np.uint8)
    pil = Image.fromarray(img_u8, "RGB").convert("RGBA")
    draw = ImageDraw.Draw(pil)
    n_holes = int(rng.integers(2, 4))  # 2 or 3 holes
    hole_xs = [int(w * (i + 1) / (n_holes + 1)) for i in range(n_holes)]
    hole_y  = h // 2 + int(rng.integers(-h // 8, h // 8 + 1))
    hole_r  = int(rng.integers(8, 16))
    for hx in hole_xs:
        # worn ring around hole
        for r_ring in range(hole_r + 12, hole_r, -1):
            alpha = int(180 * (r_ring - hole_r) / 12)
            ring_color = (
                int(base_tone[0] * 255 * 0.55),
                int(base_tone[1] * 255 * 0.45),
                int(base_tone[2] * 255 * 0.30),
                alpha,
            )
            draw.ellipse(
                [hx - r_ring, hole_y - r_ring, hx + r_ring, hole_y + r_ring],
                fill=ring_color,
            )
        # the hole itself: transparent with dark rim
        draw.ellipse(
            [hx - hole_r, hole_y - hole_r, hx + hole_r, hole_y + hole_r],
            fill=(20, 15, 10, 255),
        )

    # ── frayed / cracked edge mask ──────────────────────────────────────────
    mask = _torn_edge_mask(rng, h, w, border=int(rng.integers(8, 20)))
    arr = np.array(pil)
    arr[..., 3] = (arr[..., 3].astype(np.float32) * mask).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


# ── public API ────────────────────────────────────────────────────────────────

_BUILDERS = {
    "paper":     _make_paper,
    "palm_leaf": _make_palm_leaf,
}


def make_background(rng, size, kind="paper", cfg=None):
    """Return an RGBA PIL Image background of the given kind.

    Args:
        rng:  numpy Generator
        size: (width, height) tuple
        kind: "paper" | "palm_leaf"
        cfg:  optional dict of extra per-kind parameters (for future use)
    """
    w, h = size
    builder = _BUILDERS.get(kind, _make_paper)
    return builder(rng, w, h, cfg)
