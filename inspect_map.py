"""Decode a real RA2 map's packs to check coordinate conventions."""
import sys, struct, collections, configparser
sys.path.insert(0, '.')
import ra2codec as C

def sections(path):
    secs, cur = {}, None
    for line in open(path, encoding='latin-1'):
        line = line.split(';')[0].strip()
        if line.startswith('['):
            cur = line[1:line.index(']')]; secs.setdefault(cur, {})
        elif '=' in line and cur:
            k, v = line.split('=', 1); secs[cur][k.strip()] = v.strip()
    return secs

if __name__ == "__main__":
    path = sys.argv[1]    
    s = sections(path)
    W, H = map(int, s['Map']['Size'].split(',')[2:])
    iso = C.unpack(C.from_ini_section(s['IsoMapPack5'].values()), 5)
    n = len(iso) // 11
    print('W,H', W, H, 'cells', n, 'expected', (2*W-1)*H, 'rem', len(iso) % 11)
    cells = [struct.unpack_from('<HHiBBB', iso, 11*k) for k in range(n)]
    rx = [c[0] for c in cells]; ry = [c[1] for c in cells]
    print('rx', min(rx), max(rx), 'ry', min(ry), max(ry))
    print('first', cells[:3], 'last', cells[-2:])
    bad = 0
    for (x, y, t, st, z, ice) in cells:
        dx = x - y + W - 1; dy = x + y - W - 1
        if not (0 <= dx <= 2*W-2 and 0 <= dy <= 2*H-1): bad += 1
    print('cells outside expected diamond', bad)
    tc = collections.Counter(c[2] for c in cells)
    print('top tiles', tc.most_common(15))
    ramps = collections.Counter((c[2], c[4]) for c in cells if 29 <= c[2] <= 48)
    print('ramp tiles used', sorted(set(c[2] for c in cells if 29 <= c[2] <= 48)))
    print('levels', collections.Counter(c[4] for c in cells))
    ov = C.unpack(C.from_ini_section(s['OverlayPack'].values()), 80)
    od = C.unpack(C.from_ini_section(s['OverlayDataPack'].values()), 80)
    print('overlay len', len(ov), len(od))
    oc = collections.Counter(b for b in ov if b != 255)
    print('overlay types', sorted(oc.items()))
    valid = {(c[0], c[1]) for c in cells}
    for name, f in [('rx+512ry', lambda i: (i % 512, i // 512)), ('ry+512rx', lambda i: (i // 512, i % 512))]:
        inside = sum(1 for i, b in enumerate(ov) if b != 255 and f(i) in valid)
        print(name, 'overlays on valid cells', inside, 'of', sum(oc.values()))
