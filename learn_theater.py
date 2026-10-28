"""Re-learn ramp / LAT / shore rules for one theater from real maps of that theater."""
import collections, glob, json, struct, sys
import ra2codec as C, tileset
from inspect_map import sections
theater, ypath = sys.argv[1], sys.argv[2]
T = tileset.load(ypath)
PAT = {0:(0,0,0,0),1:(0,1,1,0),2:(0,0,1,1),3:(1,0,0,1),4:(1,1,0,0),5:(0,0,1,0),6:(0,0,0,1),7:(1,0,0,0),8:(0,1,0,0),
       9:(0,1,1,1),10:(1,0,1,1),11:(1,1,0,1),12:(1,1,1,0),13:(0,1,2,1),14:(1,0,1,2),15:(2,1,0,1),16:(1,2,1,0),
       17:(0,1,0,1),18:(1,0,1,0),19:(0,1,0,1),20:(1,0,1,0)}
OFF = [(0,0),(1,0),(1,1),(0,1)]
N4 = [(0,-1),(1,0),(0,1),(-1,0)]
GROUND = {'Lat Grass','LAT Sand','Sand Individual','LAT Grass Rough','GrassRough Individual','LAT Grass, thick',
          'GrassThick Idividual','Pavement','LAT Pavement','Pavement Individual'}
maps = []
for p in sorted(glob.glob('ref/maps/*.map')):
    s = sections(p)
    if s.get('Map', {}).get('Theater') != theater: continue
    iso = C.unpack(C.from_ini_section(s['IsoMapPack5'].values()), 5)
    cells = {}
    for k in range(len(iso)//11):
        x,y,t,st,z,_ = struct.unpack_from('<HHiBBB', iso, 11*k)
        t = 0 if t == 65535 else t
        if t in T and st in T[t]['tiles']: cells[(x,y)] = (t, st, z)
    maps.append((p, cells))
print(len(maps), 'maps')
# 1. ramp model: observed corner pattern per ramp type from flat ground neighbours
obs = collections.defaultdict(collections.Counter)
use = collections.Counter()
for p, cells in maps:
    corner = collections.defaultdict(set)
    for (x,y),(t,st,z) in cells.items():
        use[t] += 1
        if T[t]['cat'] in GROUND and T[t]['tiles'][st]['ramp'] == 0:
            for ox,oy in OFF: corner[(x+ox,y+oy)].add(z)
    for (x,y),(t,st,z) in cells.items():
        r = T[t]['tiles'][st]['ramp']
        if T[t]['cat'] == 'Ramps':
            o = tuple((min(corner[(x+ox,y+oy)])-z) if len(corner.get((x+ox,y+oy), ())) == 1 else None for ox,oy in OFF)
            if None not in o: obs[r][o] += 1
bad = sum(n for r, c in obs.items() for o, n in c.items() if o != PAT[r]); tot = sum(sum(c.values()) for c in obs.values())
print(f'ramps: {tot} fully observed ramp cells, {bad} disagree with the model;', {r: obs[r].most_common(1)[0] for r in sorted(obs)})
# 2. LAT rule
for base, lat in [(534,463),(131,132),(418,419),(493,494)]:
    good = badl = 0
    for p, cells in maps:
        member = lambda t: t == base or lat <= t < lat+16
        for (x,y),(t,st,z) in cells.items():
            if lat <= t < lat+16:
                mask = sum((not member(cells.get((x+dx,y+dy), (base,))[0])) << i for i,(dx,dy) in enumerate(N4))
                good += t - lat == mask; badl += t - lat != mask
    print(f'LAT {base}/{lat}: good {good} bad {badl}  base color {T[base]["tiles"][0]["color"]} terrain {T[base]["tiles"][0]["terrain"]}')
# 3. shore cases
pat_of = collections.defaultdict(collections.Counter)
for p, cells in maps:
    corner = collections.defaultdict(set)
    for (x,y),(t,st,z) in cells.items():
        cat = T[t]['cat']
        if cat == 'Shore Pieces' or 'Water' in cat and cat != 'Water' or 'bridge' in cat.lower(): continue
        for ox,oy in OFF: corner[(x+ox,y+oy)].add('W' if cat == 'Water' else 'L')
    for (x,y),(t,st,z) in cells.items():
        if T[t]['cat'] == 'Shore Pieces':
            pat_of[(t,st)][''.join((lambda v: next(iter(v)) if len(v)==1 else '?')(corner.get((x+ox,y+oy), set())) for ox,oy in OFF)] += 1
cases = {}
for t in sorted({k[0] for k in pat_of}):
    w, h = T[t]['size']
    if (w, h) != (2, 2): continue
    grid = collections.defaultdict(collections.Counter)
    for st in range(4):
        for pat, n in pat_of.get((t, st), {}).items():
            for (dx, dy), ch in zip(OFF, pat):
                if ch != '?': grid[(st % 2 + dx, st // 2 + dy)][ch] += n
    lab = {k: max(v, key=v.get) for k, v in grid.items()}
    rows = [''.join(lab.get((cx, cy), '?') for cx in range(3)) for cy in range(3)]
    cases[t] = (''.join(lab.get(k, '?') for k in [(0,0),(2,0),(2,2),(0,2)]), rows, sum(sum(pat_of[(t,s)].values()) for s in range(4)))
ref = json.load(open('shore_cases.json'))
for t, (case, rows, n) in cases.items():
    flag = '' if str(t) not in ref or ref[str(t)][0] == case else f'  <-- temperate says {ref[str(t)][0]}'
    print(f'shore {t}: {case} {rows} n={n}{flag}')
print('most used tiles:', [(t, T[t]['cat'], n) for t, n in use.most_common(25)])
