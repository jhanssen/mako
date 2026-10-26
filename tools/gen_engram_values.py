"""Rewrites the decoded-value lines ("V t layer values...") of tests/data/ds41_engram_expected.txt from the Q6_K Engram
tables of the Maya-S set (shards 5 and 6): position t's first-column row of each layer, decoded from the raw table bytes
with the Q6_K block layout (float32 products), independently of mako's decoder.

    python3 tools/gen_engram_values.py tests/data/ds41_engram_expected.txt <shard 5> <data start> <shard 6> <data start>

The data start is the offset of a shard's tensor data (192 for both in the current set; gguf_dump shows the tables).
"""
import struct, sys
exp, s5, d5, s6, d6 = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
lines = open(exp).read().split('\n')
rows = {}
for ln in lines:
    p = ln.split()
    if len(p) > 2 and p[0].isdigit():
        rows[(int(p[0]), int(p[1]))] = int(p[2])
f32 = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]
def q6k(raw):
    out = [0.0] * 256
    d = struct.unpack('<e', raw[208:210])[0]
    sc = struct.unpack('<16b', raw[192:208])
    for n in range(2):
        ql = raw[64 * n:64 * n + 64]; qh = raw[128 + 32 * n:128 + 32 * n + 32]; s = sc[8 * n:8 * n + 8]
        for l in range(32):
            i = l // 16
            q = [((ql[l] & 15) | ((qh[l] & 3) << 4)) - 32, ((ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4)) - 32,
                 ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32, ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32]
            for j in range(4):
                out[128 * n + l + 32 * j] = f32(d * s[i + 2 * j] * q[j])
    return out
files = [(open(s5, 'rb'), d5), (open(s6, 'rb'), d6)]
out = [ln for ln in lines if not ln.startswith('V ')]
while out and out[-1] == '':
    out.pop()
for t in range(4):
    for l in range(2):
        fh, base = files[l]
        fh.seek(base + rows[(t, l)] * 210)
        out.append('V %d %d ' % (t, l) + ' '.join(repr(v) for v in q6k(fh.read(210))))
open(exp, 'w').write('\n'.join(out) + '\n')
print('wrote', sum(1 for x in out if x.startswith('V ')), 'value rows')
