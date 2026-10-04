#!/usr/bin/env python3
"""Synthetic Indic manuscript generator.

Usage:
    python generate.py                       # uses config.yaml, 100 images per script
    python generate.py --scripts devanagari --n 5
    python generate.py --push                # also upload to Hugging Face
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from src import backgrounds, deform, ink, layout, render, annotate, text_source


def split_names(n, ratios):
    n_val  = round(n * ratios["validation"])
    n_test = round(n * ratios["test"])
    return (["train"] * (n - n_val - n_test)
            + ["validation"] * n_val
            + ["test"]       * n_test)


def _scfg_get(scfg, global_render, key):
    """Return per-script override if present, else global render default."""
    return scfg.get(key, global_render.get(key))


def make_sample(cfg, script_cfg, corpus, rng):
    """Build one complete manuscript folio image + ground-truth lines."""
    size          = tuple(cfg["image_size"])          # (W, H)
    global_render = cfg.get("render", {})
    deform_cfg    = cfg.get("deform", {})

    # ── font / corpus params (per-script overrides respected) ────────────────
    font_path  = script_cfg["font"]
    base_size  = int(_scfg_get(script_cfg, global_render, "base_font_size") or 34)
    main_words = int(_scfg_get(script_cfg, global_render, "main_words")     or 140)
    note_words = int(_scfg_get(script_cfg, global_render, "note_words")     or 30)

    # ── background ───────────────────────────────────────────────────────────
    bg_kinds = cfg["background_kinds"]
    kind     = bg_kinds[int(rng.integers(len(bg_kinds)))]
    bg       = backgrounds.make_background(rng, size, kind).convert("RGBA")

    # ── page layout ──────────────────────────────────────────────────────────
    lay = layout.build_layout(rng, size)

    # ── text passages ─────────────────────────────────────────────────────────
    text_main = text_source.sample_passage(rng, corpus, main_words)
    text_note = text_source.sample_passage(rng, corpus, note_words)

    # ── compose all text layers ──────────────────────────────────────────────
    text_layer, blocks = render.compose_text_layers(
        size, text_main, text_note, font_path, lay, rng,
        base_size=base_size,
    )

    # ── ink aging artifacts ──────────────────────────────────────────────────
    text_layer = ink.apply_artifacts(text_layer, rng)

    # ── composite: background + text ────────────────────────────────────────
    page = Image.alpha_composite(bg, text_layer).convert("RGB")

    # ── physical deformations ────────────────────────────────────────────────
    page = deform.warp(
        page, rng,
        amp      = float(deform_cfg.get("warp_amp",  6.0)),
        do_curl  = bool (deform_cfg.get("do_curl",   True)),
        do_folds = bool (deform_cfg.get("do_folds",  True)),
    )

    return page, blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  default="config.yaml")
    ap.add_argument("--scripts", nargs="*", help="subset of scripts to generate")
    ap.add_argument("--n",       type=int,  help="override samples_per_script")
    ap.add_argument("--push",    action="store_true", help="upload to Hugging Face")
    args = ap.parse_args()

    cfg      = yaml.safe_load(open(args.config, encoding="utf-8"))
    n        = args.n or cfg["samples_per_script"]
    out_root = Path(cfg["output_dir"])

    for script, scfg in cfg["scripts"].items():
        if args.scripts and script not in args.scripts:
            continue

        corpus = text_source.load_corpus(scfg["corpus"])
        font_ok = Path(scfg["font"]).exists()

        if not corpus or not font_ok:
            print(f"[skip] {script}: missing corpus ({scfg['corpus']}) "
                  f"or font ({scfg['font']})")
            continue

        rng    = np.random.default_rng(cfg["seed"] + sum(map(ord, script)))
        splits = split_names(n, cfg["splits"])
        rng.shuffle(splits)
        meta   = {s: [] for s in ("train", "validation", "test")}

        for i, split in enumerate(tqdm(splits, desc=script), start=1):
            d = out_root / script / split
            d.mkdir(parents=True, exist_ok=True)

            img, blocks = make_sample(cfg, scfg, corpus, rng)
            name = f"Image_{i}"
            img.save(d / f"{name}.png")
            annotate.write_md(d / f"{name}.md", blocks)

            # flatten all block lines for metadata
            all_lines = [ln for block in blocks for ln in block]
            meta[split].append({
                "file_name": f"{name}.png",
                "text":      "\n".join(all_lines),
                "script":    script,
            })

        for split, rows in meta.items():
            if rows:
                path = out_root / script / split / "metadata.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)

    if args.push:
        from src.export_hf import push
        push(out_root, cfg["hf_repo"])


if __name__ == "__main__":
    main()
