#!/usr/bin/env python3
"""Build an archival contact sheet from generated manuscript folios."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def create_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    cols: int = 5,
    thumb_w: int = 420,
    thumb_h: int = 300,
    title: str = "Indic Manuscript Synthesis — Realism Contact Sheet",
):
    if not image_paths:
        print("[!] No images found for contact sheet.")
        return None

    n = len(image_paths)
    rows = (n + cols - 1) // cols
    pad_x = 24
    pad_y = 28
    header_h = 70
    footer_h = 30

    sheet_w = cols * thumb_w + (cols + 1) * pad_x
    sheet_h = header_h + rows * thumb_h + (rows + 1) * pad_y + footer_h

    sheet = Image.new("RGB", (sheet_w, sheet_h), (28, 26, 24))
    draw = ImageDraw.Draw(sheet)

    try:
        font_title = ImageFont.truetype("fonts/NotoSansDevanagari-Regular.ttf", 26)
        font_sub   = ImageFont.truetype("fonts/NotoSansDevanagari-Regular.ttf", 15)
        font_badge = ImageFont.truetype("fonts/NotoSansDevanagari-Regular.ttf", 13)
    except Exception:
        font_title = font_sub = font_badge = ImageFont.load_default()

    # Draw header
    draw.text((pad_x, 16), title, font=font_title, fill=(240, 230, 210))
    sub_text = "5-Sample Inspection: Palm Leaf & Handmade Paper | Scribe Profile & Re-Dip Fading | Rubricated Lines | Verse Markers | Smudges & Blots"
    draw.text((pad_x, 48), sub_text, font=font_sub, fill=(180, 170, 155))

    for idx, img_path in enumerate(image_paths):
        r = idx // cols
        c = idx % cols
        x = pad_x + c * (thumb_w + pad_x)
        y = header_h + pad_y + r * (thumb_h + pad_y)

        # Load and resize thumbnail maintaining aspect ratio
        try:
            im = Image.open(img_path).convert("RGB")
            im.thumbnail((thumb_w - 8, thumb_h - 32), Image.Resampling.LANCZOS)

            # Draw matte background for folio
            mat_w = thumb_w
            mat_h = thumb_h
            draw.rectangle([x, y, x + mat_w, y + mat_h], fill=(42, 38, 35), outline=(65, 58, 52), width=1)

            # Paste centered inside matte
            px = x + (thumb_w - im.width) // 2
            py = y + 8 + (thumb_h - 40 - im.height) // 2
            sheet.paste(im, (px, py))

            # Script badge & filename label
            script_name = img_path.parent.parent.name if img_path.parent.parent != Path("output") else img_path.parent.name
            label = f"{script_name.upper()} • {img_path.name}"
            draw.rectangle([x + 6, y + mat_h - 26, x + mat_w - 6, y + mat_h - 6], fill=(22, 20, 18))
            draw.text((x + 12, y + mat_h - 24), label, font=font_badge, fill=(215, 195, 165))
        except Exception as e:
            print(f"[!] Error placing {img_path}: {e}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)
    print(f"Contact sheet saved to: {output_path}")
    return output_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="output", help="Output directory")
    ap.add_argument("--sheet_name", default="contact_sheet.png", help="Contact sheet filename")
    ap.add_argument("--n", type=int, default=5, help="Number of samples to include")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    # Collect all generated png files in train split
    all_imgs = sorted(out_dir.glob("*/*/*.png"))

    if not all_imgs:
        all_imgs = sorted(out_dir.glob("*/*.png"))

    # For the 5-sample primary contact sheet:
    # Pick up to 5 images across the scripts or from devanagari/modi/sharada
    selected_5 = []
    scripts = sorted(list({p.parent.parent.name for p in all_imgs if p.parent.parent.name != "output"}))

    # Take samples evenly across available scripts up to 5
    for s in scripts:
        s_imgs = [p for p in all_imgs if p.parent.parent.name == s]
        if s_imgs:
            selected_5.append(s_imgs[0])
    # Fill up to 5
    for p in all_imgs:
        if p not in selected_5 and len(selected_5) < args.n:
            selected_5.append(p)

    out_path = out_dir / args.sheet_name
    create_contact_sheet(selected_5[:args.n], out_path, cols=min(args.n, 5))

    # Also build a comprehensive contact sheet of all generated samples
    if len(all_imgs) > args.n:
        all_path = out_dir / "contact_sheet_all.png"
        create_contact_sheet(
            all_imgs, all_path,
            cols=5,
            title="Complete Multi-Script Synthetic Folios Overview"
        )


if __name__ == "__main__":
    main()
