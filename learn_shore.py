import struct, collections, glob, json
import ra2codec as C, tileset
from inspect_map import sections
T = tileset.load()
OFF = [(0,0),(1,0),(1,1),(0,1)]   # corners: top, right, bottom, left
WATERCAT = {'Water'}
SHORE = {'Shore Pieces'}
IGNORE = {'Cliff/Water pieces', 'Water bridge', 'Bridges', 'Wood Bridges', 'Waterfalls', 'Waterfalls-B', 'Waterfalls-C', 'Waterfalls-D'}
pat_of = collections.defaultdict(collections.Counter)
zrel = collections.Counter()
for p in glob.glob('ref/maps/*.map'):
    s = sections(p)
    if s['Map'].get('Theater') != 'TEMPERATE': continue
    iso = C.unpack(C.from_ini_section(s['IsoMapPack5'].values()), 5)
    cells = {}
    for k in range(len(iso)//11):
        x,y,t,st,z,_ = struct.unpack_from('<HHiBBB', iso, 11*k); cells[(x,y)] = (0 if t == 65535 else t, st, z)
    corner = collections.defaultdict(set)
    for (x,y),(t,st,z) in cells.items():
        cat = T[t]['cat']
        if cat in SHORE or cat in IGNORE: continue
        kind = 'W' if cat in WATERCAT else 'L'
        for ox,oy in OFF: corner[(x+ox,y+oy)].add(kind)
    for (x,y),(t,st,z) in cells.items():
        if T[t]['cat'] in SHORE:
            pat = ''.join((lambda v: v.pop() if len(v)==1 else '?')(set(corner.get((x+ox,y+oy), set()))) for ox,oy in OFF)
            pat_of[(t,st)][pat] += 1
            # level relative to adjacent water
            wz = [cells[(x+dx,y+dy)][2] for dx,dy in ((0,-1),(1,0),(0,1),(-1,0)) if (x+dx,y+dy) in cells and T[cells[(x+dx,y+dy)][0]]['cat'] in WATERCAT]
            if wz: zrel[z - wz[0]] += 1
print('shore z - water z', zrel)
good = collections.defaultdict(collections.Counter)
amb = 0
for key, c in pat_of.items():
    full = {p: n for p, n in c.items() if '?' not in p}
    if not full: continue
    best, n = max(full.items(), key=lambda kv: kv[1])
    tot = sum(full.values())
    if n / tot < 0.8: amb += 1
    good[best][key] += n
print('ambiguous subtiles', amb, 'of', len(pat_of))
for p in sorted(good): print(p, good[p].most_common(6))
print('---- partial patterns per template/subtile (top 2)')
for key in sorted(pat_of):
    t, st = key
    print(key, T[t]['size'], pat_of[key].most_common(3))
print('---- template corner grids (rows = cy, cols = cx); block-corner case = (0,0),(2,0),(2,2),(0,2)')
OUTCASE = {}
for t in sorted({k[0] for k in pat_of}):
    w, h = T[t]['size']
    grid = collections.defaultdict(collections.Counter)
    for st in range(w * h):
        ox, oy = st % w, st // w
        for pat, n in pat_of.get((t, st), {}).items():
            for (dx, dy), ch in zip(OFF, pat):
                if ch != '?': grid[(ox+dx, oy+dy)][ch] += n
    lab = {k: (max(v, key=v.get) if v[max(v, key=v.get)] >= 0.9*sum(v.values()) else '*') for k, v in grid.items()}
    rows = [''.join(lab.get((cx, cy), '?') for cx in range(w+1)) for cy in range(h+1)]
    case = ''.join(lab.get(k, '?') for k in [(0,0),(w,0),(w,h),(0,h)])
    print(t, (w, h), rows, 'case', case)
    if (w, h) == (2, 2): OUTCASE[t] = (case, rows)
json.dump(OUTCASE, open('shore_cases.json', 'w'))
