#!/usr/bin/env python3
"""Copies a GGUF with one string metadata value replaced; the header is rewritten and the tensor data copied as is.

    gguf_retag.py <in.gguf> <out.gguf> <key> <value>

Tensor offsets count from the aligned start of the data section, so only the header changes.
"""
import shutil
import struct
import sys

SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def main() -> int:
    if len(sys.argv) != 5:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    src, dst, key, value = sys.argv[1:5]
    with open(src, "rb") as f:
        head = f.read(64 << 20)
        pos = 0

        def take(fmt):
            nonlocal pos
            v = struct.unpack_from("<" + fmt, head, pos)[0]
            pos += struct.calcsize("<" + fmt)
            return v

        def string():
            nonlocal pos
            n = take("Q")
            pos += n
            return head[pos - n:pos]

        def skip(t):
            nonlocal pos
            if t == 8:
                string()
            elif t == 9:
                et, n = take("I"), take("Q")
                for _ in range(n):
                    skip(et)
            else:
                pos += SIZES[t]

        if head[:4] != b"GGUF":
            print(f"{src}: not a GGUF file", file=sys.stderr)
            return 1
        pos = 4
        take("I")
        n_tensors, n_kv = take("Q"), take("Q")
        out = bytearray(head[:24])
        align, found = 32, False
        for _ in range(n_kv):
            start = pos
            k = string()
            t = take("I")
            if k == b"general.alignment":
                align = struct.unpack_from("<I", head, pos)[0]
            skip(t)
            if k.decode() == key:
                if t != 8:
                    print(f"{key} is not a string", file=sys.stderr)
                    return 1
                v = value.encode()
                out += struct.pack("<Q", len(k)) + k + struct.pack("<IQ", 8, len(v)) + v
                found = True
            else:
                out += head[start:pos]
        if not found:
            print(f"{src}: no {key}", file=sys.stderr)
            return 1
        infos = pos
        for _ in range(n_tensors):
            string()
            n_dims = take("I")
            pos += 8 * n_dims + 4 + 8
        out += head[infos:pos]
        data = (pos + align - 1) // align * align
        out += b"\0" * (-len(out) % align)
        with open(dst, "wb") as o:
            o.write(out)
            f.seek(data)
            shutil.copyfileobj(f, o, 64 << 20)
    return 0


if __name__ == "__main__":
    sys.exit(main())
