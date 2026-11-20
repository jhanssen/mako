#!/usr/bin/env python3
"""Converts per-stream refusal directions for Hy4-flash to an ablation file (mtl::Ablation's MAKOABL1 layout).

    python3 tools/hy4f_ablation.py <in.bin> <out.abl>

The input: int32 points, streams, k, width, then float32 [points][streams][k][width], point l being the streams
entering layer l and the last point the streams before the output head. The output has the sites stream0..3 (points
0 to layers - 1) and head0..3 (the last point, at the last layer's index; zero elsewhere).
"""
import struct
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    data = open(sys.argv[1], "rb").read()
    points, streams, k, width = struct.unpack("<4i", data[:16])
    if streams != 4 or k <= 0 or width <= 0 or len(data) != 16 + points * streams * k * width * 4:
        sys.exit(f"{sys.argv[1]}: header {points} x {streams} x {k} x {width} does not match its {len(data)} bytes")
    layers = points - 1
    vec = k * width * 4
    zero = bytes(vec)

    def at(point, stream):
        o = 16 + (point * streams + stream) * vec
        return data[o:o + vec]

    sites = [f"stream{s}" for s in range(4)] + [f"head{s}" for s in range(4)]
    with open(sys.argv[2], "wb") as f:
        f.write(b"MAKOABL1" + struct.pack("<4I", width, layers, len(sites), k))
        for s in sites:
            f.write(s.encode().ljust(16, b"\0"))
        for l in range(layers):
            for s in range(4):
                f.write(at(l, s))
            for s in range(4):
                f.write(at(layers, s) if l == layers - 1 else zero)
    print(f"{sys.argv[2]}: {layers} layers x {len(sites)} sites x {k} directions x {width}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
