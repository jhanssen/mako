"""Writes an ablation file (mtl::Ablation's MAKOABL1 layout) from raw float32 directions [n_layer][k][n_embd], the same
directions at every site given.

    python3 tools/make_ablation.py <raw.f32> <out> --layers N --embd E --sites ffn_out,stream

(GLM-5.3's sites: ffn_out, stream.) k follows from the raw file's size.
"""
import argparse
import os
import struct
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--embd", type=int, required=True)
    ap.add_argument("--sites", required=True, help="comma-separated site names")
    a = ap.parse_args()
    sites = a.sites.split(",")
    size = os.path.getsize(a.raw)
    per_k = a.layers * a.embd * 4
    if size % per_k:
        sys.exit(f"{a.raw}: {size} bytes are not {a.layers} layers x k x {a.embd} float32")
    k = size // per_k
    if any(len(s.encode()) > 16 for s in sites):
        sys.exit("site names are at most 16 bytes")
    data = open(a.raw, "rb").read()
    layer_bytes = k * a.embd * 4
    with open(a.out, "wb") as f:
        f.write(b"MAKOABL1" + struct.pack("<4I", a.embd, a.layers, len(sites), k))
        for s in sites:
            f.write(s.encode().ljust(16, b"\0"))
        for l in range(a.layers):
            for _ in sites:
                f.write(data[l * layer_bytes:(l + 1) * layer_bytes])
    print(f"{a.out}: {a.layers} layers x {len(sites)} sites x {k} directions x {a.embd}")


if __name__ == "__main__":
    main()
