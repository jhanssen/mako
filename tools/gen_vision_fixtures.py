"""Writes the test images of tests/data/vision: smooth ramps plus a periodic checker (compresses well), in the common
PNG colour types and bit depths, and RGB at sizes that take each sizing path.

    python3 tools/gen_vision_fixtures.py tests/data/vision
"""
import struct, sys, zlib
def png(path, w, h, ctype, depth, rows, plte=None):
    def chunk(t, d): return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    bpp = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype] * depth // 8
    def filtered(i, r):  # the smaller of Sub and Up by sum of residuals
        sub = bytes((r[k] - (r[k - bpp] if k >= bpp else 0)) % 256 for k in range(len(r)))
        up = bytes((r[k] - (rows[i - 1][k] if i else 0)) % 256 for k in range(len(r)))
        cost = lambda f: sum(v if v < 128 else 256 - v for v in f)
        return b'\1' + sub if cost(sub) <= cost(up) else b'\2' + up
    raw = b''.join(filtered(i, r) for i, r in enumerate(rows))
    out = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, depth, ctype, 0, 0, 0))
    if plte: out += chunk(b'PLTE', plte)
    out += chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b'')
    open(path, 'wb').write(out)
def val(x, y, c, w, h):
    ramp = (x * 255 // max(1, w - 1), y * 255 // max(1, h - 1), (x + y) * 255 // max(1, w + h - 2))[c]
    return (ramp + ((x // 5 + y // 3) % 2) * 60) % 256
def rgb(w, h): return [bytes(val(x, y, c, w, h) for x in range(w) for c in range(3)) for y in range(h)]
d = sys.argv[1]
for w, h in ((1024, 768), (200, 150), (300, 1200), (3200, 2400), (517, 333)):
    png(f'{d}/rgb_{w}x{h}.png', w, h, 2, 8, rgb(w, h))
w, h = 333, 222
png(f'{d}/rgba8.png', w, h, 6, 8, [bytes(([val(x, y, c, w, h) for c in range(3)] + [(x * 3 + y) % 256])[i]
                                         for x in range(w) for i in range(4)) for y in range(h)])
png(f'{d}/gray8.png', w, h, 0, 8, [bytes(val(x, y, 0, w, h) for x in range(w)) for y in range(h)])
png(f'{d}/graya8.png', w, h, 4, 8, [bytes(v for x in range(w) for v in (val(x, y, 1, w, h), (x + y) % 256)) for y in range(h)])
png(f'{d}/rgb16.png', w, h, 2, 16, [b''.join(struct.pack('>H', val(x, y, c, w, h) * 256 + (x // 7 + y) % 256)
                                            for x in range(w) for c in range(3)) for y in range(h)])
pal = bytes(v for i in range(256) for v in ((i * 3) % 256, (i * 7) % 256, 255 - i))
png(f'{d}/pal8.png', w, h, 3, 8, [bytes((x // 3 + y // 2) % 256 for x in range(w)) for y in range(h)], plte=pal)
