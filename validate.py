"""Read the generated .mpr back and check it against the rules learned from real RA2 maps."""
import collections, struct, sys
import ra2codec as C, tileset
from inspect_map import sections
T = tileset.load()
path = sys.argv[1]
raw = open(path, 'rb').read()
assert b'\r\n' in raw and raw.count(b'\n') == raw.count(b'\r\n'), 'line endings'
s = sections(path)
W, H = map(int, s['Map']['Size'].split(',')[2:])
iso = C.unpack(C.from_ini_section(s['IsoMapPack5'].values()), 5)
assert len(iso) == (2 * W - 1) * H * 11, len(iso)
cells = {}
for k in range(len(iso) // 11):
    x, y, t, st, z, ice = struct.unpack_from('<HHiBBB', iso, 11 * k)
    dx, dy = x - y + W - 1, x + y - W - 1
    assert 0 <= dx <= 2 * W - 2 and 0 <= dy <= 2 * H - 1, (x, y)
    assert t in T and st in T[t]['tiles'], t
    assert (x, y) not in cells
    cells[(x, y)] = (t, st, z)
print('cells ok', len(cells))
# ramps: corner heights must agree between all cells sharing a corner
PAT = {0:(0,0,0,0),1:(0,1,1,0),2:(0,0,1,1),3:(1,0,0,1),4:(1,1,0,0),5:(0,0,1,0),6:(0,0,0,1),7:(1,0,0,0),8:(0,1,0,0),
       9:(0,1,1,1),10:(1,0,1,1),11:(1,1,0,1),12:(1,1,1,0),13:(0,1,2,1),14:(1,0,1,2),15:(2,1,0,1),16:(1,2,1,0),
       17:(0,1,0,1),18:(1,0,1,0),19:(0,1,0,1),20:(1,0,1,0)}
corner = collections.defaultdict(set)
for (x, y), (t, st, z) in cells.items():
    for (ox, oy), d in zip([(0,0),(1,0),(1,1),(0,1)], PAT[T[t]['tiles'][st]['ramp']]):
        corner[(x + ox, y + oy)].add(z + d)
print('corner conflicts', sum(len(v) > 1 for v in corner.values()))
# LAT rule
N4 = [(0,-1),(1,0),(0,1),(-1,0)]
bad = 0
for base, lat in [(534,463),(131,132),(418,419),(493,494)]:
    member = lambda t: t == base or lat <= t < lat + 16
    for (x, y), (t, st, z) in cells.items():
        if lat <= t < lat + 16 or t == base:
            mask = sum((not member(cells.get((x+dx, y+dy), (base,))[0])) << i for i, (dx, dy) in enumerate(N4) if (x+dx, y+dy) in cells)
            bad += (t - lat if t != base else 0) != mask
print('LAT mismatches', bad)
ov = C.unpack(C.from_ini_section(s['OverlayPack'].values()), 80)
od = C.unpack(C.from_ini_section(s['OverlayDataPack'].values()), 80)
assert len(ov) == len(od) == 262144
ovc = collections.Counter()
for i, b in enumerate(ov):
    if b != 255:
        x, y = i % 512, i // 512
        assert (x, y) in cells and cells[(x, y)][0] not in range(29, 49), (x, y)
        assert 1 <= od[i] <= 11
        ovc[b] += 1
print('overlays', dict(ovc))
trees = {}
for k, v in s['Terrain'].items():
    x, y = int(k) % 1000, int(k) // 1000
    assert (x, y) in cells, k
    assert ov[x + 512 * y] == 255, ('tree on ore', k)
    trees[(x, y)] = v
print('terrain objects', collections.Counter(trees.values()).most_common(3), len(trees))
for k, v in s['Structures'].items():
    f = v.split(',')
    assert len(f) == 17, v
    x, y = int(f[3]), int(f[4])
    assert all(cells[(x + a, y + b)][0] not in range(29, 49) for a in range(2) for b in range(2)), v
    print('  structure', f[1], x, y, 'level', cells[(x, y)][2])
for k, v in s['Waypoints'].items():
    v = int(v); x, y = v % 1000, v // 1000
    print('  start', k, (x, y), cells[(x, y)])
prev = C.unpack(C.from_ini_section(s['PreviewPack'].values()), 5)
pw, ph = map(int, s['Preview']['Size'].split(',')[2:])
assert len(prev) == pw * ph * 3 and pw == 2 * W and ph == H
print('header', s['Header'])
print('basic', {k: s['Basic'][k] for k in ('Name', 'MaxPlayer', 'MinPlayer', 'MultiplayerOnly', 'GameMode')})
print('ALL CHECKS PASSED')
