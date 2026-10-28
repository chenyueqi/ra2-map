import sys, struct, collections, itertools
import ra2codec as C, tileset
from inspect_map import sections
T = tileset.load()
# corner order: top(rx,ry) right(rx+1,ry) bottom(rx+1,ry+1) left(rx,ry+1)
PAT = {0:(0,0,0,0),1:(0,1,1,0),2:(0,0,1,1),3:(1,0,0,1),4:(1,1,0,0),5:(0,0,1,0),6:(0,0,0,1),7:(1,0,0,0),8:(0,1,0,0),
       9:(0,1,1,1),10:(1,0,1,1),11:(1,1,0,1),12:(1,1,1,0),13:(0,1,2,1),14:(1,0,1,2),15:(2,1,0,1),16:(1,2,1,0),
       17:(0,1,0,1),18:(1,0,1,0),19:(0,1,0,1),20:(1,0,1,0)}
OFF = [(0,0),(1,0),(1,1),(0,1)]
s = sections(sys.argv[1])
iso = C.unpack(C.from_ini_section(s['IsoMapPack5'].values()), 5)
cells = [struct.unpack_from('<HHiBBB', iso, 11*k) for k in range(len(iso)//11)]
def ramp(t, st):
    t = 0 if t in (65535, -1) else t
    tl = T.get(t, {}).get('tiles', {}).get(st)
    return tl['ramp'] if tl else None
def score(pat):
    corners = collections.defaultdict(set)
    for x, y, t, st, z, _ in cells:
        r = ramp(t, st)
        if r is None or r not in pat: continue
        for (ox, oy), d in zip(OFF, pat[r]): corners[(x+ox, y+oy)].add(z+d)
    return sum(1 for v in corners.values() if len(v) > 1), len(corners)
print('my model conflicts', score(PAT))
# alternative: rotate pattern assignment by k corners to see if another orientation fits better
for k in range(1,4):
    alt = {r: p[k:]+p[:k] for r,p in PAT.items()}
    print('rotated', k, score(alt))
print('mirrored', score({r:(p[0],p[3],p[2],p[1]) for r,p in PAT.items()}))
