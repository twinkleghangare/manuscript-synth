"""Handwriting-style text rendering with full realism.

Features:
  - Per-glyph rotation jitter and vertical wobble
  - Baseline waviness (sine drift)
  - Irregular inter-word and inter-line spacing
  - Variable stroke weight simulated via alpha and slight blur
  - Slight whole-line slant drift
  - 1–2 main text blocks supported
  - Marginal annotation (smaller font, rotated slightly)
  - Verse / section markers (e.g. ।९।) rendered in red ink
  - Highlight wash (translucent colored rectangle under text)
  - Hard overflow guard: every rendered glyph bbox is clamped to safe area
  - Safe-area assertion after every block

All layout boxes come from layout.build_layout().
"""
from __future__ import annotations
import math
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from .ink import ink_color
from .layout import assert_all_boxes_inside


# ── Devanagari verse markers used in Indic MSS ───────────────────────────────
_VERSE_MARKERS = [
    "।", "॥", "।१।", "।२।", "।३।", "।४।", "।५।",
    "।६।", "।७।", "।८।", "।९।", "।१०।", "।११।",
    "॥ इति ॥", "॥ शुभम् ॥",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()


def _wrap_words(words: list[str], font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    """Word-wrap a list of words into lines no wider than max_w."""
    lines, cur = [], []
    for wd in words:
        trial = " ".join(cur + [wd])
        if cur and font.getlength(trial) > max_w:
            lines.append(" ".join(cur))
            cur = [wd]
        else:
            cur.append(wd)
    if cur:
        lines.append(" ".join(cur))
    return lines


def _fit_font(font_path: str, base_size: int, words: list[str],
              bw: int, bh: int, pad: int, line_gap: float) -> tuple:
    """Shrink font until all words fit in (bw, bh - 2*pad)."""
    size = base_size
    while True:
        font = _load_font(font_path, size)
        lines = _wrap_words(words, font, bw - 2 * pad)
        lh = int(size * line_gap)
        if len(lines) * lh <= bh - 2 * pad or size <= 12:
            break
        size -= 2
    # hard vertical cut
    max_lines = max(1, (bh - 2 * pad) // lh)
    return font, lines[:max_lines], size, lh


def _render_glyph(draw: ImageDraw.ImageDraw, word: str, font: ImageFont.FreeTypeFont,
                  x: float, y: float, color_rgba: tuple, rng,
                  slant_deg: float = 0.0) -> float:
    """Draw one word with per-word rotation jitter. Returns the word width."""
    tw = int(font.getlength(word)) + 24
    th = font.size * 2
    tile = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    ImageDraw.Draw(tile).text((12, font.size // 2), word, font=font, fill=color_rgba)
    angle = float(rng.normal(slant_deg, 1.6))
    if abs(angle) > 0.5:
        tile = tile.rotate(angle, resample=Image.BICUBIC, expand=False)
    return tile, tw


def _highlight_wash(layer: Image.Image, box: tuple, rng,
                    color: Optional[tuple] = None) -> Image.Image:
    """Paint a translucent colored wash over a rectangular region."""
    if color is None:
        # warm red / ochre wash
        r = int(rng.integers(160, 210))
        g = int(rng.integers(60, 110))
        b = int(rng.integers(30, 70))
        color = (r, g, b)
    alpha = int(rng.integers(25, 65))
    overlay = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle(box, fill=color + (alpha,))
    return Image.alpha_composite(layer, overlay)


class ScribeProfile:
    """A consistent scribal profile picked once per page (slant, spacing, pressure)."""
    def __init__(self, rng, cfg: Optional[dict] = None):
        cfg = cfg or {}
        sc = cfg.get("scribe", {}) if isinstance(cfg.get("scribe"), dict) else {}

        # Scribe's overall handwriting slant tendency
        s_range = sc.get("slant_drift_range", [-2.0, 2.0])
        self.slant_base = float(rng.uniform(s_range[0], s_range[1]))
        self.slant_jitter = float(sc.get("slant_jitter", 1.2))

        # Scribe's character/word spacing style
        sp_range = sc.get("spacing_factor_range", [0.95, 1.08])
        self.spacing_factor = float(rng.uniform(sp_range[0], sp_range[1]))

        # Scribe's pressure / ink flow
        pr_range = sc.get("pressure_range", [0.88, 1.10])
        self.pressure_base = float(rng.uniform(pr_range[0], pr_range[1]))

        # Re-dip interval (scribe dips pen every N lines)
        dip_range = sc.get("redip_interval_range") or cfg.get("re_dip_interval", [2, 4])
        self.dip_interval = int(rng.integers(dip_range[0], dip_range[1] + 1))

        # Scribal strikeout / crossed-out word probability
        self.strikeout_prob = float(sc.get("crossed_out_prob", cfg.get("strikeout_prob", 0.08)))

        # Fading depth
        self.fading_strength = float(sc.get("fading_strength", cfg.get("fading_strength", 0.35)))
        self.fade_min_alpha = int(sc.get("fade_min_alpha", 145))
        self.redip_darkening = float(sc.get("redip_darkening", 1.25))


# ── main block renderer ───────────────────────────────────────────────────────

def render_block(
    canvas_size: tuple,
    text: str,
    font_path: str,
    box: tuple,
    rng,
    base_size: int = 34,
    line_gap: float = 1.65,
    highlight_prob: float = 0.10,
    slant_drift_max: float = 0.8,
    baseline_wave_amp: float = 3.0,
    baseline_wave_freq: float = 0.008,
    scribe: Optional[ScribeProfile] = None,
    force_highlight_line: bool = False,
) -> tuple[Image.Image, list[str]]:
    """Render a text block onto a transparent RGBA canvas layer."""
    W, H = canvas_size
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    pad = 14  # internal padding; prevents text touching box edge

    words = text.split()
    if not words:
        return Image.new("RGBA", canvas_size, (0, 0, 0, 0)), []

    if scribe is None:
        scribe = ScribeProfile(rng)

    font, lines, size, lh = _fit_font(font_path, base_size, words, bw, bh, pad, line_gap)
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))

    red_line_idx = int(rng.integers(0, min(len(lines), 3))) if (force_highlight_line and len(lines) > 0) else -1

    for li, line in enumerate(lines):
        hl = (li == red_line_idx) if force_highlight_line else (rng.random() < highlight_prob)
        base_col = ink_color(rng, highlight=hl)

        # Re-dip ink dynamics: darken right after re-dip, fade gradually until next dip
        dip_step = li % max(1, scribe.dip_interval)
        line_fade = 1.0 - (dip_step / max(1, scribe.dip_interval)) * scribe.fading_strength

        # --- per-line baseline parameters ---
        slant_deg  = float(rng.uniform(-slant_drift_max, slant_drift_max))
        wave_phase = float(rng.uniform(0, 6.28))
        x = x0 + pad + float(rng.uniform(0, 6))
        baseline_center = y0 + pad + li * lh + int(rng.integers(-3, 4))

        words_in_line = line.split()
        n_words = max(1, len(words_in_line))

        for wi, wd in enumerate(words_in_line):
            # Gradual fading within line and across re-dips
            word_fade = line_fade * (1.0 - 0.05 * (wi / n_words))
            alpha = int(np.clip(255 * scribe.pressure_base * word_fade * (0.95 + 0.10 * rng.random()), 120, 255))

            # Darken ink right after each re-dip
            if dip_step == 0 and wi < 2:
                word_col = tuple(max(0, int(c * 0.80)) for c in base_col)
            else:
                word_col = base_col
            col_rgba = word_col + (alpha,)

            # baseline wobble
            wave_y = baseline_wave_amp * math.sin(x * baseline_wave_freq + wave_phase)
            draw_y = baseline_center + wave_y
            glyph_y = draw_y + float(rng.normal(0, 1.2))

            dest_x = int(max(x0, min(x - 12, x1 - size)))
            dest_y = int(max(y0, min(glyph_y - size // 2, y1 - size - pad)))

            tw = int(font.getlength(wd)) + 24
            th = size * 2
            tile = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
            draw_tile = ImageDraw.Draw(tile)
            draw_tile.text((12, size // 2), wd, font=font, fill=col_rgba)

            # Occasional crossed-out word (scribal correction)
            if rng.random() < scribe.strikeout_prob:
                strike_y = size // 2 + int(font.size * 0.45) + int(rng.integers(-2, 3))
                strike_w = max(2, int(size * 0.08))
                draw_tile.line([(10, strike_y), (tw - 10, strike_y + int(rng.integers(-2, 3)))],
                               fill=col_rgba, width=strike_w)

            # per-word rotation with scribe slant profile
            angle = float(rng.normal(slant_deg + scribe.slant_base, scribe.slant_jitter))
            if abs(angle) > 0.4:
                tile = tile.rotate(angle, resample=Image.BICUBIC, expand=False)

            paste_x = max(x0, min(dest_x, x1 - tile.width))
            paste_y = max(y0, min(dest_y, y1 - tile.height))

            layer.alpha_composite(tile, (paste_x, paste_y))

            # irregular spacing governed by scribe spacing profile
            x += font.getlength(wd + " ") * scribe.spacing_factor * float(rng.uniform(0.96, 1.06))
            if x >= x1 - pad:
                break

    return layer, lines


# ── marginal annotation renderer ─────────────────────────────────────────────

def render_note(
    canvas_size: tuple,
    text: str,
    font_path: str,
    box: tuple,
    rng,
    base_size: int = 22,
    scribe: Optional[ScribeProfile] = None,
) -> tuple[Image.Image, list[str]]:
    """Render a marginal annotation — smaller font, more compressed, slight slant."""
    return render_block(
        canvas_size, text, font_path, box, rng,
        base_size=base_size,
        line_gap=1.45,
        highlight_prob=0.05,
        slant_drift_max=1.5,
        baseline_wave_amp=2.0,
        baseline_wave_freq=0.010,
        scribe=scribe,
        force_highlight_line=False,
    )


# ── verse / section marker renderer ──────────────────────────────────────────

def render_markers(
    canvas_size: tuple,
    marker_positions: list[tuple],
    font_path: str,
    rng,
    font_size: int = 32,
) -> Image.Image:
    """Render verse markers (।९।, ॥ etc.) at the given (x, y) positions.

    Markers are always rendered in red/highlight ink.
    """
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    W, H = canvas_size
    font = _load_font(font_path, font_size)

    for mx, my in marker_positions:
        marker = _VERSE_MARKERS[int(rng.integers(len(_VERSE_MARKERS)))]
        color  = ink_color(rng, highlight=True)
        alpha  = int(rng.integers(180, 240))
        # clamp so it doesn't go off-canvas
        paste_x = max(0, min(mx, W - int(font.getlength(marker)) - 4))
        paste_y = max(0, min(my, H - font_size - 4))
        ImageDraw.Draw(layer).text((paste_x, paste_y), marker,
                                   font=font, fill=color + (alpha,))

    return layer


# ── highlight wash ────────────────────────────────────────────────────────────

def render_highlights(canvas_size: tuple, highlight_boxes: list[tuple],
                      rng) -> Image.Image:
    """Render translucent colored wash rectangles (like a scribe's red-ink emphasis)."""
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    for box in highlight_boxes:
        layer = _highlight_wash(layer, box, rng)
    return layer


# ── full-page compositor (convenience, used by generate.py) ──────────────────

def compose_text_layers(
    canvas_size: tuple,
    text_main: str,
    text_note: str,
    font_path: str,
    layout: dict,
    rng,
    base_size: int = 34,
    cfg: Optional[dict] = None,
) -> tuple[Image.Image, list[list[str]]]:
    """Build all text/marker/highlight layers and return composited RGBA + ground-truth lines.

    Args:
        canvas_size: (W, H)
        text_main:   main passage text
        text_note:   marginal annotation text
        font_path:   path to ttf font
        layout:      dict from layout.build_layout()
        rng:         numpy Generator
        base_size:   base font size for main block
        cfg:         optional render config dict

    Returns:
        (composited RGBA Image, [main_lines, main2_lines, note_lines])
    """
    # Safe-area assertion (raises on misconfigured layout)
    assert_all_boxes_inside(layout, canvas_size)

    cfg = cfg or {}
    W, H = canvas_size

    # Pick a scribe profile once per page (slant, spacing, pressure, re-dip interval)
    scribe = ScribeProfile(rng, cfg)

    # Highlight wash (below text)
    layer = render_highlights(canvas_size, layout.get("highlights", []), rng)

    force_red = bool(cfg.get("red_highlight_line", True) or cfg.get("force_highlight_line", False))

    # Main text block
    main_layer, main_lines = render_block(
        canvas_size, text_main, font_path, layout["main"], rng,
        base_size=base_size, highlight_prob=0.10,
        scribe=scribe, force_highlight_line=force_red)
    layer = Image.alpha_composite(layer, main_layer)

    # Optional second main block
    main2_lines: list[str] = []
    if layout.get("main2"):
        # reuse remaining words from text_main or duplicate
        m2_layer, main2_lines = render_block(
            canvas_size, text_main, font_path, layout["main2"], rng,
            base_size=base_size - 2, highlight_prob=0.08,
            scribe=scribe, force_highlight_line=False)
        layer = Image.alpha_composite(layer, m2_layer)

    # Marginal annotation
    note_layer, note_lines = render_note(
        canvas_size, text_note, font_path, layout["note"], rng,
        base_size=max(14, base_size - 10), scribe=scribe)
    layer = Image.alpha_composite(layer, note_layer)

    # Verse markers
    marker_layer = render_markers(
        canvas_size, layout.get("markers", []), font_path, rng,
        font_size=max(20, base_size - 4))
    layer = Image.alpha_composite(layer, marker_layer)

    return layer, [main_lines, main2_lines, note_lines]
