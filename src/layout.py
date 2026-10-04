"""Page layout engine.

Produces a layout dict with named boxes, all guaranteed inside the safe area:
  - "main"     : primary text block (always present)
  - "main2"    : optional second text block below the first (50 % chance)
  - "note"     : marginal annotation strip (always present, left or right)
  - "markers"  : list of (x, y) positions for verse/section markers like ।९।
  - "highlight" : list of (x0, y0, x1, y1) sub-rects for colored-wash highlights

Safe-area guarantee: every returned box satisfies
    margin ≤ x0 < x1 ≤ W - margin
    margin ≤ y0 < y1 ≤ H - margin

A standalone test function assert_all_boxes_inside() is provided.
"""
from __future__ import annotations
from typing import Any


# ── helpers ───────────────────────────────────────────────────────────────────

def _clamp(box, W, H, margin):
    """Clamp a box tuple to the safe area."""
    x0, y0, x1, y1 = box
    x0 = max(margin, min(x0, W - margin))
    y0 = max(margin, min(y0, H - margin))
    x1 = max(x0 + 2, min(x1, W - margin))
    y1 = max(y0 + 2, min(y1, H - margin))
    return (x0, y0, x1, y1)


def _valid(box, W, H, margin):
    x0, y0, x1, y1 = box
    return (x0 >= margin and y0 >= margin
            and x1 <= W - margin and y1 <= H - margin
            and x1 > x0 + 10 and y1 > y0 + 10)


# ── public: layout builder ────────────────────────────────────────────────────

def build_layout(rng, size, margin: int = 90) -> dict[str, Any]:
    """Return a layout dict with boxes for text rendering.

    All returned (x0, y0, x1, y1) boxes are inside the safe margin.
    """
    W, H = size

    # ── marginal strip (left or right) ───────────────────────────────────────
    margin_w = int(rng.integers(130, 210))
    note_gap  = int(rng.integers(20, 40))
    side = "left" if rng.random() < 0.5 else "right"

    if side == "right":
        main_box = (margin, margin, W - margin - margin_w - note_gap, H - margin)
        note_box = (W - margin - margin_w, margin + 30,
                    W - margin,            H - margin - 30)
    else:
        note_box = (margin,              margin + 30,
                    margin + margin_w,   H - margin - 30)
        main_box = (margin + margin_w + note_gap, margin,
                    W - margin,                   H - margin)

    main_box = _clamp(main_box, W, H, margin)
    note_box = _clamp(note_box, W, H, margin)

    layout: dict[str, Any] = {
        "main": main_box,
        "note": note_box,
        "main2": None,
        "markers": [],
        "highlights": [],
    }

    # ── optional second text block ────────────────────────────────────────────
    if rng.random() < 0.50:
        mx0, my0, mx1, my1 = main_box
        block_split = int(rng.uniform(0.45, 0.65))  # fraction of height for block1
        gap = int(rng.integers(15, 35))
        mid_y = my0 + int((my1 - my0) * block_split)
        main1 = _clamp((mx0, my0, mx1, mid_y - gap // 2), W, H, margin)
        main2 = _clamp((mx0, mid_y + gap // 2, mx1, my1), W, H, margin)
        if _valid(main1, W, H, margin) and _valid(main2, W, H, margin):
            layout["main"]  = main1
            layout["main2"] = main2

    # ── verse / section markers ───────────────────────────────────────────────
    # place 2–5 markers along the right edge of the main block or between lines
    mx0, my0, mx1, my1 = layout["main"]
    n_markers = int(rng.integers(2, 6))
    for _ in range(n_markers):
        mx = int(rng.uniform(mx0 + (mx1 - mx0) * 0.60, mx1 - 10))
        my = int(rng.uniform(my0 + 10, my1 - 10))
        layout["markers"].append((mx, my))

    # ── highlight bands ───────────────────────────────────────────────────────
    # 0–3 short horizontal highlight washes within the main block
    n_hl = int(rng.integers(0, 4))
    for _ in range(n_hl):
        hx0 = mx0 + int(rng.uniform(0, (mx1 - mx0) * 0.25))
        hx1 = mx1 - int(rng.uniform(0, (mx1 - mx0) * 0.25))
        hy0 = int(rng.uniform(my0 + 20, my1 - 40))
        hy1 = hy0 + int(rng.integers(20, 45))
        hl = _clamp((hx0, hy0, hx1, hy1), W, H, margin)
        if _valid(hl, W, H, margin):
            layout["highlights"].append(hl)

    return layout


# ── test / assertion helper ───────────────────────────────────────────────────

def assert_all_boxes_inside(layout: dict, size: tuple[int, int],
                             margin: int = 90) -> None:
    """Assert that every box in the layout dict is within the safe area.

    Raises AssertionError with details on the first failing box.
    """
    W, H = size

    def _check(name, box):
        x0, y0, x1, y1 = box
        assert x0 >= margin,     f"layout '{name}' x0={x0} < margin={margin}"
        assert y0 >= margin,     f"layout '{name}' y0={y0} < margin={margin}"
        assert x1 <= W - margin, f"layout '{name}' x1={x1} > W-margin={W - margin}"
        assert y1 <= H - margin, f"layout '{name}' y1={y1} > H-margin={H - margin}"
        assert x1 > x0,         f"layout '{name}' zero-width box"
        assert y1 > y0,         f"layout '{name}' zero-height box"

    for key in ("main", "note"):
        if layout.get(key):
            _check(key, layout[key])

    if layout.get("main2"):
        _check("main2", layout["main2"])

    for i, hl in enumerate(layout.get("highlights", [])):
        _check(f"highlight[{i}]", hl)
