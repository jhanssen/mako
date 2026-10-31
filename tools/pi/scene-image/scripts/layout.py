#!/usr/bin/env python3
"""Turns a layout draft (boxes in pixels from the top-left) into a MOSS-Image layout prompt.

    layout.py DRAFT        prints WIDTHxHEIGHT on the first line, the prompt (one line of JSON) on the second
DRAFT is a file, "-" for standard input, or the JSON itself (starting with "{").

The draft: {"width": W, "height": H, "summary": {"caption": "...", "meta": [...], "canvas": "single"},
            "items": [{"label": "character" | "object" | "text", "caption": "...", "bbox": [x0, y0, x1, y1],
                       "text": "..." (text items), "tags": [...], "palette": [...]}, ...]}
The prompt's boxes are [y0, x0, y1, x1] from the image's centre, edges rounded outward.
"""
import json, math, sys

ITEM_KEYS = {"label", "caption", "bbox", "text", "tags", "palette"}
SUMMARY_KEYS = {"caption", "meta", "canvas", "palette"}


def fail(msg):
    sys.exit(f"layout: {msg}")


def main():
    if len(sys.argv) != 2:
        fail("usage: layout.py DRAFT (a file, - for standard input, or the JSON)")
    src = sys.argv[1]
    try:
        if src == "-":
            d = json.load(sys.stdin)
        elif src.lstrip().startswith("{"):
            d = json.loads(src)
        else:
            d = json.load(open(src))
    except Exception as e:
        fail(f"not a JSON draft: {e}")
    if not isinstance(d, dict):
        fail("the draft must be a JSON object")
    w, h = d.get("width"), d.get("height")
    if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0 and w % 32 == 0 and h % 32 == 0):
        fail("width and height must be positive multiples of 32")
    s = d.get("summary")
    if not isinstance(s, dict) or not isinstance(s.get("caption"), str) or not s["caption"].strip():
        fail("summary.caption must describe the whole image")
    if set(s) - SUMMARY_KEYS:
        fail(f"summary: unknown fields {sorted(set(s) - SUMMARY_KEYS)}")
    items = d.get("items")
    if not isinstance(items, list) or not items:
        fail("items must be a nonempty list")
    out = []
    for i, it in enumerate(items):
        if not isinstance(it, dict) or set(it) - ITEM_KEYS:
            fail(f"item {i}: unknown fields {sorted(set(it) - ITEM_KEYS) if isinstance(it, dict) else it}")
        if it.get("label") not in ("character", "object", "text"):
            fail(f"item {i}: label must be character, object or text")
        if not isinstance(it.get("caption"), str) or not it["caption"].strip():
            fail(f"item {i}: needs a caption")
        if it["label"] == "text" and not isinstance(it.get("text"), str):
            fail(f"item {i}: a text item needs its text")
        b = it.get("bbox")
        if not (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)):
            fail(f"item {i}: bbox must be [x0, y0, x1, y1]")
        x0, y0, x1, y1 = b
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            fail(f"item {i}: bbox {b} is not inside the {w}x{h} image")
        o = {k: it[k] for k in ("label", "caption", "text", "tags", "palette") if k in it}
        o["bbox"] = [math.floor(y0) - h // 2, math.floor(x0) - w // 2, math.ceil(y1) - h // 2, math.ceil(x1) - w // 2]
        out.append(o)
    layout = {"summary": s, "items": out, "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    print(f"{w}x{h}")
    print(json.dumps(layout, ensure_ascii=False, separators=(",", ":")))


main()
