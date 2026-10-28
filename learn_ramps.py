import sys, struct, collections
import ra2codec as C, tileset
from inspect_map import sections
T = tileset.load()
OFF = [(0,0),(1,0),(1,1),(0,1)]
OK={'Lat Grass','LAT Sand','Sand Individual','LAT Grass Rough','GrassRough Individual','LAT Grass, thick','GrassThick Idividual','Pavement','LAT Pavement','Pavement Individual'}
stats = collections.defaultdict(collections.Counter)
for path in sys.argv[1:]:
    s = sections(path)
    iso = C.unpack(C.from_ini_section(s['IsoMapPack5'].values()), 5)
    cells = [struct.unpack_from('<HHiBBB', iso, 11*k) for k in range(len(iso)//11)]
    corner = collections.defaultdict(set)
    for x,y,t,st,z,_ in cells:
        t = 0 if t == 65535 else t
        if T[t]['cat'] in OK and T[t]['tiles'][st]['ramp'] == 0:
            for ox,oy in OFF: corner[(x+ox,y+oy)].add(z)
    for x,y,t,st,z,_ in cells:
        if 29 <= t <= 48:
            obs = []
            for ox,oy in OFF:
                v = corner.get((x+ox,y+oy))
                obs.append(min(v)-z if v and len(v)==1 else '?')
            stats[t-28][tuple(obs)] += 1
for r in sorted(stats): print(r, stats[r].most_common(4))
