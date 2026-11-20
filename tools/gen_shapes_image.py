#!/usr/bin/env python3
"""Writes tests/data/vision/shapes.png: three shapes in known places and a line of text, for questions about an image.

    gen_shapes_image.py <out.png>

Needs Pillow; the text uses Arial from macOS (Pillow's default font without it, which changes the image).
"""
import sys

from PIL import Image, ImageDraw, ImageFont


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    im = Image.new("RGB", (800, 600), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.ellipse((40, 40, 220, 220), fill=(220, 30, 30))  # red circle, top left
    d.rectangle((580, 400, 760, 560), fill=(30, 60, 220))  # blue square, bottom right
    d.polygon([(650, 40), (560, 200), (740, 200)], fill=(30, 170, 60))  # green triangle, top right
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 64)
    except OSError:
        font = ImageFont.load_default()
    d.text((230, 270), "MAKO 2026", fill=(0, 0, 0), font=font)
    im.save(sys.argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
