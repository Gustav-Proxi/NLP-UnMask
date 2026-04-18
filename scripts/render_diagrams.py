"""
Render ASCII anatomy diagrams to PNG images using PIL.
Run once: python scripts/render_diagrams.py
Outputs to public/anatomy/<concept_id>.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw, ImageFont
from src.anatomy_images import ANATOMY_DIAGRAMS

DEST = Path(__file__).parent.parent / "public" / "anatomy"
DEST.mkdir(parents=True, exist_ok=True)

BG       = (18, 18, 24)       # near-black background
FG       = (220, 220, 235)    # light text
ACCENT   = (99, 179, 237)     # blue caption
PADDING  = 20
LINE_H   = 18
FONT_SZ  = 14

def _load_font(size: int):
    candidates = [
        "/Library/Fonts/Courier New.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()

font    = _load_font(FONT_SZ)
caption_font = _load_font(FONT_SZ - 1)

for concept_id, data in ANATOMY_DIAGRAMS.items():
    out_path = DEST / f"{concept_id.replace('.', '_')}.png"
    diagram  = data["diagram"]
    caption  = data["caption"]
    lines    = diagram.splitlines()

    # Calculate canvas size
    max_chars = max((len(l) for l in lines), default=40)
    width  = max_chars * (FONT_SZ // 2 + 1) + PADDING * 2 + 10
    height = (len(lines) + 3) * LINE_H + PADDING * 3  # +3 for caption

    img  = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)

    # Caption bar
    draw.rectangle([0, 0, width, LINE_H * 2 + PADDING], fill=(30, 30, 45))
    draw.text((PADDING, PADDING // 2 + 2), f"📌 {caption}", font=caption_font, fill=ACCENT)

    # Diagram text
    y = LINE_H * 2 + PADDING + 4
    for line in lines:
        draw.text((PADDING, y), line, font=font, fill=FG)
        y += LINE_H

    img.save(out_path)
    print(f"  ok  {out_path.name}  ({width}x{height})")

print(f"\nRendered {len(ANATOMY_DIAGRAMS)} diagrams to {DEST}")
