"""Learn which cliff pieces may sit next to each other from the official temperate maps -> cliff_rules.json.

A cell is a token: 'L' (flat ground at the cliff foot), 'H' (flat ground on top, 4 levels up) or a cliff
subtile 't.s' of the temperate "Cliff Set" (templates 49-88). Cliff cells sit at foot + their tile height
(0 or 4). For each of the 4 forward neighbour offsets we record every token pair seen in the official maps
whose cells belong to one cliff tier (same foot level). Ground next to ground is not learned: L-L and H-H
are always allowed, L-H never (a ramp or a gap there would let units up the cliff).
"""
import collections, glob, json, struct
import ra2codec as C, tileset
from inspect_map import sections

T = tileset.load('ref/temperat.yaml')
CLIFF = set(range(49, 89))
DIRS = [(1, 0), (0, 1), (1, 1), (1, -1)]
pairs = [collections.Counter() for _ in DIRS]
freq = collections.Counter()
for path in sorted(glob.glob('ref/maps/*.map')):
    s = sections(path)
    if s.get('Map', {}).get('Theater') != 'TEMPERATE':
        continue
    iso = C.unpack(C.from_ini_section(s['IsoMapPack5'].values()), 5)
    cells = {}
    for k in range(len(iso) // 11):
        x, y, t, st, z, _ = struct.unpack_from('<HHiBBB', iso, 11 * k)
        cells[(x, y)] = (0 if t == 65535 else t, st, z)

    def complete(x, y):           # the whole template instance is there (pieces get overwritten in editors)
        t, st, _ = cells[(x, y)]
        w = T[t]['size'][0]
        ox, oy = x - st % w, y - st // w
        return all(cells.get((ox + i % w, oy + i // w), (None, None))[:2] == (t, i) for i in T[t]['tiles'])

    def token(c, foot):
        t, st, z = cells[c]
        if t in CLIFF:
            return f'{t}.{st}' if complete(*c) and z - T[t]['tiles'][st]['height'] == foot else None
        if T[t]['cat'] in ('Ramps', 'Slope Set Pieces', 'Ramp edge fixup') or T[t]['tiles'][st]['terrain'] == 'Cliff':
            return None
        return {foot: 'L', foot + 4: 'H'}.get(z)

    for (x, y), (t, st, z) in cells.items():
        if t not in CLIFF or not complete(x, y):
            continue
        foot = z - T[t]['tiles'][st]['height']
        freq[f'{t}.{st}'] += 1
        for d, (dx, dy) in enumerate(DIRS):       # pairs touching this cliff cell, both ways
            for a, b in (((x, y), (x + dx, y + dy)), ((x - dx, y - dy), (x, y))):
                if a in cells and b in cells:
                    ta, tb = token(a, foot), token(b, foot)
                    if ta and tb:
                        pairs[d][(ta, tb)] += 1
rules = {'dirs': DIRS, 'freq': dict(freq),
         'height': {f'{t}.{s}': v['height'] for t in sorted(CLIFF) for s, v in T[t]['tiles'].items()},
         'pairs': [sorted([a, b] for a, b in p) for p in pairs]}
json.dump(rules, open('cliff_rules.json', 'w'), indent=0)
print('cliff cells', sum(freq.values()), 'subtiles', len(freq), 'pairs per direction', [len(p) for p in pairs])
