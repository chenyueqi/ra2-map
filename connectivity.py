"""Which land masses of a generated map are connected (shore and water cells count as water)."""
import importlib, json, sys
import numpy as np
region, conn = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4
cfg = importlib.import_module(f'regions.{region}')
st = np.load(f'out/{region}/state.npz'); obj = json.load(open(f'out/{region}/objects.json'))
W, H = obj['W'], obj['H']; LX, LY, LW, LH = obj['local']
X0, X1, Y0, Y1 = -(W - 2*LX), 2*(LX+LW) - W, W + 2*LY, W + 2*(LY+LH+1)
tile, valid = st['tile'], st['valid']
water = ((tile >= 89) & (tile <= 130)) | ((tile >= 314) & (tile <= 327))
N = valid.shape[0]
CX, CY = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
land = valid & ~water & (CX-CY > X0) & (CX-CY < X1) & (CX+CY > Y0) & (CX+CY <= Y1)
import buildings
CIV = buildings.load(obj.get('theater', 'TEMPERATE'))
DIMS = {'CAOILD': (2, 2), 'CAAIRP': (3, 3), 'CAHOSP': (6, 4), 'CAOUTP': (4, 3)}
for v in obj['structures']:                     # buildings and trees block movement
    f = v.split(','); x, y = int(f[3]), int(f[4])
    w, h = DIMS.get(f[1]) or (CIV[f[1]]['w'], CIV[f[1]]['h'])
    land[x:x + w, y:y + h] = False
for k in obj['terrain']:
    land[int(k) % 1000, int(k) // 1000] = False
nb = [(1,0),(-1,0),(0,1),(0,-1)] + ([(1,1),(1,-1),(-1,1),(-1,-1)] if conn == 8 else [])
lab = np.full(land.shape, -1); sizes = []
for s in zip(*np.nonzero(land)):
    if lab[s] >= 0: continue
    stack = [s]; lab[s] = len(sizes); n = 0
    while stack:
        x, y = stack.pop(); n += 1
        for dx, dy in nb:
            a, b = x+dx, y+dy
            if land[a, b] and lab[a, b] < 0: lab[a, b] = len(sizes); stack.append((a, b))
    sizes.append(n)
def to_iso(lon, lat):
    X = X0 + (lon - cfg.LON0)/(cfg.LON1 - cfg.LON0)*(X1 - X0); Y = Y0 + (lat - cfg.LAT0)/(cfg.LAT1 - cfg.LAT0)*(Y1 - Y0)
    return int((X+Y)/2), int((Y-X)/2)
print(f'{conn}-connected land components >= 20 cells:', sorted([s for s in sizes if s >= 20], reverse=True))
for name, (x, y) in zip(obj['names'], obj['starts']):
    print(f'  start {name:16s} comp {lab[x, y]}')
for name, lon, lat in getattr(cfg, 'CHECK_POINTS', []):
    x, y = to_iso(lon, lat); c = lab[x, y]
    print(f'  {name:22s} comp {c}' + ('' if c >= 0 else ' (water/shore)'))
print('component sizes by label:', {i: s for i, s in enumerate(sizes) if s >= 20})
if len(sys.argv) > 3:
    from PIL import Image
    pal = [(230, 80, 80), (80, 200, 80), (80, 120, 240), (240, 200, 60), (200, 80, 200), (60, 200, 200)]
    img = np.zeros((Y1 - Y0 + 2, X1 - X0 + 2, 3), np.uint8)
    for x, y in zip(*np.nonzero(valid)):
        X, Y = x - y - X0, x + y - Y0
        if 0 <= X < img.shape[1] and 0 <= Y < img.shape[0]:
            img[Y, X] = (30, 40, 90) if water[x, y] else (pal[lab[x, y] % 6] if lab[x, y] >= 0 else (0, 0, 0))
    Image.fromarray(img).resize((img.shape[1] * 3, img.shape[0] * 3 // 2), Image.NEAREST).save(sys.argv[3])
ov = st['overlay']
for c, n in enumerate(sizes):
    if 20 <= n < max(sizes):
        m = lab == c
        xs, ys = np.nonzero(m)
        lon = cfg.LON0 + ((xs - ys).mean() - X0) / (X1 - X0) * (cfg.LON1 - cfg.LON0)
        lat = cfg.LAT0 + ((xs + ys).mean() - Y0) / (Y1 - Y0) * (cfg.LAT1 - cfg.LAT0)
        print(f'  comp {c}: {n} cells near ({lon:.3f}, {lat:.3f}), ore/gem cells {int((m & (ov != 255)).sum())}')
