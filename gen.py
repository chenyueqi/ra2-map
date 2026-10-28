"""Generate a Red Alert 2 multiplayer map of Colorado from real elevation and river data."""
import glob
import json
import math
import os
import random
import struct

import numpy as np
from PIL import Image, ImageDraw

import ra2codec as C
import tileset

OUT = os.path.join(os.path.dirname(__file__), 'out')
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(1876)  # Colorado statehood
random.seed(1876)
TS = tileset.load()

# --- map geometry --------------------------------------------------------------
W, H = 100, 154
LX, LY, LW, LH = 2, 4, 96, 141
N = W + H + 4                      # corner grid is N x N, indexed [rx, ry]
LON0, LON1, LAT0, LAT1 = -109.05, -102.04, 41.0, 37.0
X0, X1 = -(W - 2 * LX), 2 * (LX + LW) - W        # rx - ry range of the visible area
Y0, Y1 = W + 2 * LY, W + 2 * (LY + LH + 1)       # rx + ry range of the visible area


def geo(X, Y):
    return LON0 + (X - X0) / (X1 - X0) * (LON1 - LON0), LAT0 + (Y - Y0) / (Y1 - Y0) * (LAT1 - LAT0)


def to_iso(lon, lat):
    """Continuous (rx, ry) of a lon/lat point (cell centers sit at integer + 0.5)."""
    X = X0 + (lon - LON0) / (LON1 - LON0) * (X1 - X0)
    Y = Y0 + (lat - LAT0) / (LAT1 - LAT0) * (Y1 - Y0)
    return (X + Y) / 2, (Y - X) / 2


def cell_of(lon, lat):
    rx, ry = to_iso(lon, lat)
    return int(rx), int(ry)


def valid_cells():
    for y in range(H):
        for x in range(2 * W - 1):
            dy = 2 * y + x % 2
            rx = (x + dy) // 2 + 1
            yield rx, dy - rx + W + 1


CELLS = list(valid_cells())
VALID = np.zeros((N, N), bool)
for rx, ry in CELLS:
    VALID[rx, ry] = True
CX, CY = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
CORNER_LON, CORNER_LAT = geo(CX - CY, CX + CY)
CELL_LON, CELL_LAT = geo(CX - CY, CX + CY + 1.0)


def in_local(rx, ry, margin=0):
    X, Y = rx - ry, rx + ry
    return X0 + margin < X < X1 - margin and Y0 + margin < Y <= Y1 - margin


# --- elevation ----------------------------------------------------------------
Z, TX0, TY0 = 8, 49, 95


def load_dem():
    tiles = glob.glob(os.path.join(os.path.dirname(__file__), 'data/dem/8_*.png'))
    xs = sorted({int(t.split('_')[1]) for t in tiles})
    ys = sorted({int(t.split('_')[2].split('.')[0]) for t in tiles})
    assert xs[0] == TX0 and ys[0] == TY0
    E = np.zeros((256 * len(ys), 256 * len(xs)))
    for t in tiles:
        x, y = int(t.split('_')[1]), int(t.split('_')[2].split('.')[0])
        a = np.asarray(Image.open(t).convert('RGB'), float)
        E[(y - TY0) * 256:(y - TY0 + 1) * 256, (x - TX0) * 256:(x - TX0 + 1) * 256] = \
            a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768
    return E


def box_blur(a, r, passes=3):
    for _ in range(passes):
        for axis in (0, 1):
            p = np.pad(a, [(r + 1, r) if ax == axis else (0, 0) for ax in (0, 1)], mode='edge')
            c = np.cumsum(p, axis=axis)
            a = (np.take(c, range(2 * r + 1, c.shape[axis]), axis=axis) -
                 np.take(c, range(0, c.shape[axis] - 2 * r - 1), axis=axis)) / (2 * r + 1)
    return a


def sample(E, lon, lat):
    n = 2 ** Z
    px = ((lon + 180) / 360 * n - TX0) * 256 - 0.5
    py = ((1 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2 * n - TY0) * 256 - 0.5
    px = np.clip(px, 0, E.shape[1] - 1.001)
    py = np.clip(py, 0, E.shape[0] - 1.001)
    x0, y0 = px.astype(int), py.astype(int)
    fx, fy = px - x0, py - y0
    return (E[y0, x0] * (1 - fx) * (1 - fy) + E[y0, x0 + 1] * fx * (1 - fy) +
            E[y0 + 1, x0] * (1 - fx) * fy + E[y0 + 1, x0 + 1] * fx * fy)


DEM = load_dem()
DEM_SMOOTH = box_blur(DEM, 3)
DEM_WIDE = box_blur(DEM, 8)
DEM_TERRAIN = box_blur(DEM, 6)
ELEV_CORNER = sample(DEM_TERRAIN, CORNER_LON, CORNER_LAT)
ELEV_CELL = sample(DEM_SMOOTH, CELL_LON, CELL_LAT)
RELIEF_CELL = np.abs(sample(DEM_SMOOTH, CELL_LON, CELL_LAT) - sample(DEM_WIDE, CELL_LON, CELL_LAT))

# elevation (m) -> RA2 height level; plains are kept flat, mountains get the levels
ELEV_PTS = [1000, 1700, 1950, 2250, 2550, 2850, 3150, 3450, 3750, 4100]
LVL_PTS = [0, 0, 1, 2, 3.2, 4.6, 6.0, 7.4, 8.8, 10.2]


def shift(a, dx, dy, fill):
    out = np.full_like(a, fill)
    sx = slice(max(dx, 0), a.shape[0] + min(dx, 0))
    tx = slice(max(-dx, 0), a.shape[0] + min(-dx, 0))
    sy = slice(max(dy, 0), a.shape[1] + min(dy, 0))
    ty = slice(max(-dy, 0), a.shape[1] + min(-dy, 0))
    out[tx, ty] = a[sx, sy]
    return out


def relax_min(a):
    """a(i) = min_j a(j) + manhattan(i, j): the largest 1-Lipschitz function below a."""
    while True:
        b = a
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            b = np.minimum(b, shift(a, *d, np.inf) + 1)
        if np.array_equal(a, b):
            return a
        a = b


level_f = np.interp(ELEV_CORNER, ELEV_PTS, LVL_PTS)
heights = relax_min(np.round(level_f))

# --- places --------------------------------------------------------------------
STARTS = [  # name, lon, lat (nudged onto open ground where the real town hugs a mountain or the border)
    ('Boulder', -105.13, 40.03),
    ('Colorado Springs', -104.72, 38.86),
    ('Sterling', -103.21, 40.58),
    ('Lamar', -102.80, 38.12),
    ('Grand Junction', -108.42, 39.10),
    ('Durango', -107.86, 37.33),
    ('Steamboat Springs', -106.88, 40.50),
    ('Alamosa', -105.87, 37.50),
]
ORE_TOWNS = [('Limon', -103.69, 39.26), ('Fort Morgan', -103.80, 40.25), ('La Junta', -103.54, 37.98),
             ('Walsenburg', -104.78, 37.62), ('Montrose', -107.88, 38.48), ('Meeker', -107.91, 40.04),
             ('Granby', -105.94, 40.09), ('Salida', -106.00, 38.53), ('Cortez', -108.59, 37.35),
             ('Trinidad', -104.50, 37.17), ('Burlington', -102.40, 39.30), ('Pueblo', -104.61, 38.27),
             ('Gunnison', -106.93, 38.55), ('Craig', -107.55, 40.52), ('Fort Collins', -105.08, 40.58),
             ('Kremmling', -106.39, 40.06), ('Glenwood Springs', -107.32, 39.55)]
GEM_TOWNS = [('Leadville', -106.29, 39.25), ('Cripple Creek', -105.18, 38.75), ('Silverton', -107.66, 37.81),
             ('Central City', -105.51, 39.80), ('Telluride', -107.81, 37.94)]
OIL = [('Greeley', -104.70, 40.42), ('Wattenberg', -104.85, 40.17), ('Rangely', -108.80, 40.09),
       ('Rifle', -107.78, 39.53), ('Trinidad field', -104.30, 37.30), ('Aneth', -108.85, 37.30),
       ('Akron', -103.21, 40.16), ('Kit Carson', -102.79, 38.76)]
TECH = [('CAAIRP', 'Denver International Airport', -104.67, 39.86, 3, 3),
        ('CAHOSP', 'Denver Health', -104.99, 39.73, 6, 4),
        ('CAOUTP', 'Pueblo Chemical Depot', -104.33, 38.32, 4, 3)]
CITIES = [('Denver', -104.99, 39.74, 5.5), ('Boulder', -105.13, 40.03, 3.5), ('Colorado Springs', -104.72, 38.86, 4),
          ('Fort Collins', -105.08, 40.58, 2.5), ('Pueblo', -104.61, 38.27, 2.5), ('Greeley', -104.70, 40.42, 2),
          ('Grand Junction', -108.42, 39.10, 3), ('Durango', -107.86, 37.33, 2.5), ('Alamosa', -105.87, 37.50, 2.5),
          ('Sterling', -103.21, 40.58, 2.5), ('Lamar', -102.80, 38.12, 2.5), ('Steamboat Springs', -106.88, 40.50, 2.5)]
ROADS = {  # tree-free corridors along the real highways
    'I-70': [(-104.99, 39.74), (-105.51, 39.74), (-106.07, 39.63), (-106.37, 39.64), (-107.32, 39.55),
             (-107.78, 39.53), (-108.42, 39.10), (-109.2, 39.1)],
    'US-50': [(-102.80, 38.12), (-103.54, 37.98), (-104.61, 38.27), (-105.24, 38.44), (-106.0, 38.53),
              (-106.33, 38.50), (-106.93, 38.55), (-107.88, 38.48), (-108.42, 39.10)],
    'US-160': [(-104.78, 37.62), (-105.2, 37.60), (-105.87, 37.50), (-106.8, 37.48), (-107.01, 37.27),
               (-107.86, 37.33), (-108.59, 37.35)],
    'US-40': [(-104.99, 39.74), (-105.78, 39.80), (-105.94, 40.09), (-106.39, 40.06), (-106.6, 40.38),
              (-106.88, 40.50), (-107.55, 40.52), (-108.80, 40.09)],
    'US-285': [(-104.99, 39.74), (-105.5, 39.45), (-106.0, 39.22), (-106.08, 38.50), (-106.14, 38.09),
               (-105.87, 37.50)],
    'US-550': [(-107.88, 38.48), (-107.67, 38.02), (-107.66, 37.81), (-107.86, 37.33)],
    'I-25': [(-105.08, 40.58), (-104.99, 39.74), (-104.72, 38.86), (-104.61, 38.27), (-104.78, 37.62),
             (-104.50, 37.17)],
    'I-76': [(-104.99, 39.74), (-103.80, 40.25), (-103.21, 40.58)],
}


def dist_to_point(lon, lat):
    rx, ry = to_iso(lon, lat)
    return np.hypot(CX + 0.5 - rx, CY + 0.5 - ry)


# --- flatten building ground ------------------------------------------------------
lock = np.full((N, N), np.nan)


def lock_disc(lon, lat, r):
    rx, ry = to_iso(lon, lat)
    d = np.hypot(CX - rx, CY - ry)
    area = d <= r + 0.75
    lvl = float(np.round(np.median(heights[d <= max(r * 0.6, 2)])))
    near = lock[d <= r + 5]
    if (~np.isnan(near)).any():         # join a neighbouring flat area instead of stepping next to it
        lvl = float(np.round(np.nanmedian(near)))
    lock[area & np.isnan(lock)] = lvl
    return lvl


for _, lon, lat in STARTS:
    lock_disc(lon, lat, 12)
for _, lon, lat in ORE_TOWNS:
    lock_disc(lon, lat, 4.5)
for _, lon, lat in GEM_TOWNS:
    lock_disc(lon, lat, 3.5)
for _, lon, lat in OIL:
    lock_disc(lon, lat, 3)
for _, _, lon, lat, w, h in TECH:
    lock_disc(lon, lat, max(w, h) + 1.5)

upper = relax_min(np.where(np.isnan(lock), np.inf, lock))
lower = -relax_min(np.where(np.isnan(lock), np.inf, -lock))
assert (lower <= upper).all(), 'flattened areas too close for their level difference'
heights = np.clip(heights, lower, upper).astype(int)

# --- cells: level + ramp ------------------------------------------------------------
RAMP_OF = {(0, 0, 0, 0): 0, (0, 1, 1, 0): 1, (0, 0, 1, 1): 2, (1, 0, 0, 1): 3, (1, 1, 0, 0): 4,
           (0, 0, 1, 0): 5, (0, 0, 0, 1): 6, (1, 0, 0, 0): 7, (0, 1, 0, 0): 8,
           (0, 1, 1, 1): 9, (1, 0, 1, 1): 10, (1, 1, 0, 1): 11, (1, 1, 1, 0): 12,
           (0, 1, 2, 1): 13, (1, 0, 1, 2): 14, (2, 1, 0, 1): 15, (1, 2, 1, 0): 16,
           (0, 1, 0, 1): 17, (1, 0, 1, 0): 18}
RAMP_TILE0 = 28  # slope01.tem (ramp type 1) is tile 29

level = np.zeros((N, N), int)
ramp = np.zeros((N, N), int)
for rx, ry in CELLS:
    c = (heights[rx, ry], heights[rx + 1, ry], heights[rx + 1, ry + 1], heights[rx, ry + 1])
    z = min(c)
    level[rx, ry] = z
    ramp[rx, ry] = RAMP_OF[tuple(v - z for v in c)]
flat = VALID & (ramp == 0)

# --- noise --------------------------------------------------------------------------


def value_noise(scale, seed):
    g = np.random.default_rng(seed).random((N // scale + 3, N // scale + 3))
    img = Image.fromarray((g * 255).astype(np.uint8)).resize(((N // scale + 3) * scale,) * 2, Image.BICUBIC)
    return np.asarray(img, float)[:N, :N] / 255


NOISE = 0.6 * value_noise(6, 1) + 0.4 * value_noise(3, 2)
NOISE2 = value_noise(4, 3)

# --- rivers ---------------------------------------------------------------------------
RIVERS = {'Arkansas', 'Colorado', 'Rio Grande', 'South Platte', 'San Juan', 'North Platte', 'Green',
          'Animas', 'Dolores', 'Gunnison', 'White', 'Yampa', 'Purgatoire', 'Republican',
          'S. Fork Republican', 'Arikaree', 'Big Sandy', 'Little Snake', 'Laramie', 'Cimarron', 'Taylor'}


def polyline_distance(lines, step=0.25):
    """Distance (in cells) from each cell center to the nearest polyline, by stamping sample points."""
    pts = []
    for line in lines:
        for (a, b), (c, d) in zip(line, line[1:]):
            p, q = to_iso(a, b), to_iso(c, d)
            n = max(1, int(math.hypot(q[0] - p[0], q[1] - p[1]) / step))
            pts += [(p[0] + (q[0] - p[0]) * k / n, p[1] + (q[1] - p[1]) * k / n) for k in range(n + 1)]
    dist = np.full((N, N), np.inf)
    for px, py in pts:
        x0, y0 = int(px) - 3, int(py) - 3
        if x0 < -3 or y0 < -3 or x0 >= N or y0 >= N:
            continue
        xs, ys = slice(max(x0, 0), min(x0 + 7, N)), slice(max(y0, 0), min(y0 + 7, N))
        d = np.hypot(CX[xs, ys] + 0.5 - px, CY[xs, ys] + 0.5 - py)
        dist[xs, ys] = np.minimum(dist[xs, ys], d)
    return dist


river_lines = []
for f in ('data/rivers.geojson', 'data/rivers_na.geojson'):
    for ft in json.load(open(os.path.join(os.path.dirname(__file__), f)))['features']:
        g = ft['geometry']
        if not g or ft['properties'].get('name') not in RIVERS:
            continue
        for line in (g['coordinates'] if g['type'] == 'MultiLineString' else [g['coordinates']]):
            if any(-110 <= x <= -101 and 36.5 <= y <= 41.5 for x, y in line):
                river_lines.append(line)
RIVER_D = polyline_distance(river_lines)
ROAD_D = polyline_distance(ROADS.values())

# --- ground cover -------------------------------------------------------------------
GRASS, THICK, ROUGH, SAND, PAVE = 0, 131, 418, 493, 534
LAT_START = {THICK: 132, ROUGH: 419, SAND: 494, PAVE: 463}

cover = np.full((N, N), GRASS)
forest = (ELEV_CELL > 2350) & (ELEV_CELL < 3500) & ((RELIEF_CELL > 25) | (ELEV_CELL > 2650))
riparian = RIVER_D < 1.3
cover[forest & (NOISE > 0.3)] = THICK
cover[riparian & (ELEV_CELL < 2400)] = THICK
cover[ELEV_CELL >= 3500] = ROUGH
east = CELL_LON > -104.6
cover[east & ~riparian & (NOISE > 0.58)] = SAND
cover[east & ~riparian & (NOISE2 > 0.72) & (NOISE <= 0.58)] = ROUGH
west_desert = (CELL_LON < -107.6) & (ELEV_CELL < 2000) & ~riparian
cover[west_desert & (NOISE > 0.42)] = SAND
cover[west_desert & (NOISE <= 0.42) & (NOISE2 > 0.55)] = ROUGH
slv = (CELL_LAT > 37.1) & (CELL_LAT < 38.3) & (CELL_LON > -106.3) & (CELL_LON < -105.5) & (ELEV_CELL < 2500)
cover[slv & ~riparian & (NOISE > 0.5)] = SAND
cover[dist_to_point(-105.55, 37.74) < 3.2 + 1.5 * NOISE2] = SAND        # Great Sand Dunes
for _, lon, lat, r in CITIES:
    cover[dist_to_point(lon, lat) < r + 1.6 * (NOISE2 - 0.5)] = PAVE
cover[~flat] = GRASS

tile = np.zeros((N, N), int)
for rx, ry in CELLS:
    if not flat[rx, ry]:
        tile[rx, ry] = RAMP_TILE0 + ramp[rx, ry]
        continue
    s = cover[rx, ry]
    if s == GRASS:
        continue
    mask = 0
    for bit, (dx, dy) in enumerate(((0, -1), (1, 0), (0, 1), (-1, 0))):
        nx, ny = rx + dx, ry + dy
        same = not (0 <= nx < N and 0 <= ny < N) or not VALID[nx, ny] or (flat[nx, ny] and cover[nx, ny] == s)
        mask |= (not same) << bit
    tile[rx, ry] = s if mask == 0 else LAT_START[s] + mask

# --- overlays: ore & gems ---------------------------------------------------------------
ORE, ORE_B, GEM, GEM_B = 102, 112, 27, 30
overlay = np.full((N, N), 255)
taken = np.zeros((N, N), bool)          # cells reserved by buildings / start areas


def blob(cx, cy, r):
    d = np.hypot(CX + 0.5 - cx, CY + 0.5 - cy) / (r * (0.8 + 0.4 * NOISE2))
    return d <= 1


def place_field(cx, cy, r, kind):
    m = blob(cx, cy, r) & flat & ~taken & VALID
    if kind == 'ore':
        overlay[m] = np.where(rng.random(m.sum()) < 0.85, ORE, ORE_B)
    else:
        overlay[m] = np.where(rng.random(m.sum()) < 0.85, GEM, GEM_B)
    return int(m.sum())


start_cells = []
center = to_iso(-105.55, 39.0)
for name, lon, lat in STARTS:
    rx, ry = to_iso(lon, lat)
    start_cells.append((int(rx), int(ry)))
    taken |= np.hypot(CX + 0.5 - rx, CY + 0.5 - ry) < 5.5
for (name, lon, lat), (sx, sy) in zip(STARTS, start_cells):
    away = math.atan2(sy + 0.5 - center[1], sx + 0.5 - center[0])
    placed = 0
    for da in (0.9, -0.9, 0.3, -0.3, 1.6, -1.6, 2.4, -2.4):
        if placed == 2:
            break
        a = away + da
        fx, fy = sx + 0.5 + 8.5 * math.cos(a), sy + 0.5 + 8.5 * math.sin(a)
        if not in_local(fx, fy, 6):
            continue
        if (overlay[int(fx) - 3:int(fx) + 4, int(fy) - 3:int(fy) + 4] != 255).any():
            continue
        n = place_field(fx, fy, 2.9, 'ore')
        if n >= 18:
            placed += 1
    assert placed == 2, name
for _, lon, lat in ORE_TOWNS:
    place_field(*to_iso(lon, lat), 2.6, 'ore')
for _, lon, lat in GEM_TOWNS:
    place_field(*to_iso(lon, lat), 2.0, 'gem')

# --- structures (neutral tech) ----------------------------------------------------------
structures = []


def place_building(kind, lon, lat, w, h):
    rx, ry = to_iso(lon, lat)
    x, y = int(rx - w / 2), int(ry - h / 2)
    foot = (slice(x, x + w), slice(y, y + h))
    assert flat[foot].all() and VALID[foot].all(), kind
    taken[x - 1:x + w + 1, y - 1:y + h + 1] = True
    overlay[x - 1:x + w + 1, y - 1:y + h + 1] = 255
    structures.append(f'Neutral,{kind},256,{x},{y},64,None,1,0,1,0,0,None,None,None,0,0')


for _, lon, lat in OIL:
    place_building('CAOILD', lon, lat, 2, 2)
for kind, _, lon, lat, w, h in TECH:
    place_building(kind, lon, lat, w, h)

# ore density: fuller in the middle of a field, as FinalAlert does it
odata = np.zeros((N, N), int)
has = overlay != 255
cnt = sum(shift(has.astype(int), dx, dy, 0) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy)
odata[has] = np.clip(cnt[has] + 3, 3, 11)

# --- trees ------------------------------------------------------------------------------
MOUNTAIN_TREES = ['TREE25', 'TREE26', 'TREE27', 'TREE05', 'TREE06', 'TREE07', 'TREE08', 'TREE15']
VALLEY_TREES = ['TREE01', 'TREE02', 'TREE03', 'TREE04', 'TREE10', 'TREE11', 'TREE12', 'TREE17']
p = np.zeros((N, N))
p[forest] = 0.12 + 0.45 * NOISE[forest]
p[riparian] = np.maximum(p[riparian], 0.28)
p[(ELEV_CELL >= 3500)] = 0.03
p[(p == 0)] = 0.006
p[ROAD_D < 1.2] = 0
p[cover == PAVE] = 0.0
p[~np.isnan(lock)] *= 0.15
for sx, sy in start_cells:
    p[np.hypot(CX - sx, CY - sy) < 13] = 0
p[taken | (overlay != 255) | (shift((overlay != 255).astype(int), 1, 0, 0) > 0) |
  (shift((overlay != 255).astype(int), 0, 1, 0) > 0)] = 0
terrain = {}
for rx, ry in CELLS:
    if in_local(rx, ry, 2) and rng.random() < p[rx, ry]:
        trees = MOUNTAIN_TREES if forest[rx, ry] or ELEV_CELL[rx, ry] > 2300 else VALLEY_TREES
        terrain[ry * 1000 + rx] = trees[int(rng.integers(len(trees)))]
for name, lon, lat in GEM_TOWNS[:2]:     # ore drills at the old gold camps
    rx, ry = to_iso(lon, lat)
    terrain[(int(ry) - 3) * 1000 + int(rx) + 3] = 'TIBTRE01'

# --- preview ---------------------------------------------------------------------------


def hexrgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)


def cell_color(rx, ry):
    t = tile[rx, ry]
    col = hexrgb(TS[t]['tiles'][0]['color'])
    if overlay[rx, ry] != 255:
        col = hexrgb('8C7AE6') if overlay[rx, ry] < 100 else hexrgb('D9B13B')
    elif (ry * 1000 + rx) in terrain:
        col = col * 0.55
    return np.clip(col * (0.82 + 0.035 * level[rx, ry]), 0, 255)


prev = np.zeros((H, 2 * W, 3), np.uint8)
for rx, ry in CELLS:
    dx, dy = rx - ry + W - 1, rx + ry - W - 1
    prev[dy // 2, dx] = cell_color(rx, ry)
prev[:, 2 * W - 1] = prev[:, 2 * W - 2]

# --- [Header] as FinalAlert writes it ----------------------------------------------------


def calc_pos(x, y):
    x2 = ((x * 256 + 128) - (y * 256 + 128)) * 30
    y2 = ((x * 256 + 128) + (y * 256 + 128)) * 15
    return int((math.trunc(x2 / 256) + 512 * 30) / 60), (y2 // 256) // 30


def header():
    xs, ys = [], []
    for x, y in CELLS:
        h = level[x, y]
        if (x + y > W + 2 * LY + h and x + y <= W + 2 * (LY + LH + 1) + h and
                x - y < 2 * (LX + LW) - W and y - x < W - 2 * LX):
            px, py = calc_pos(x, y)
            xs.append(px)
            ys.append(py)
    lines = [f'Width={max(xs) - min(xs)}', f'Height={max(ys) - min(ys)}', f'StartX={min(xs)}', f'StartY={min(ys)}']
    for i in range(8):
        px, py = calc_pos(*start_cells[i]) if i < len(start_cells) else (0, 0)
        lines.append(f'Waypoint{i + 1}={px},{py}')
    lines.append(f'NumberStartingPoints={len(start_cells)}')
    return lines


# --- write the map ------------------------------------------------------------------------
COUNTRY = {'Americans': 'Gold', 'Alliance': 'Gold', 'French': 'Gold', 'Germans': 'Gold', 'British': 'Gold',
           'Africans': 'Gold', 'Arabs': 'Gold', 'Confederation': 'Gold', 'Russians': 'Red',
           'GDI': 'Gold', 'Nod': 'Red', 'Neutral': 'Grey', 'Special': 'Grey'}


def build_ini():
    iso = bytearray()
    for rx, ry in sorted(CELLS, key=lambda c: (c[1], c[0])):
        iso += struct.pack('<HHiBBB', rx, ry, int(tile[rx, ry]), 0, int(level[rx, ry]), 0)
    ov = bytearray([255]) * (512 * 512)
    od = bytearray(512 * 512)
    for rx, ry in CELLS:
        if overlay[rx, ry] != 255:
            ov[rx + 512 * ry] = int(overlay[rx, ry])
            od[rx + 512 * ry] = int(odata[rx, ry])
    secs = []
    secs.append(('Header', header()))
    secs.append(('Preview', [f'Size=0,0,{2 * W},{H}']))
    secs.append(('PreviewPack', C.to_ini_lines(C.pack(prev.tobytes(), 5))))
    secs.append(('Basic', ['Name=Colorado', 'Author=Yueqi Chen', 'Percent=0', 'GameMode=standard',
                           'HomeCell=98', 'InitTime=10000', 'Official=no', 'EndOfGame=no', 'FreeRadar=no',
                           'MaxPlayer=8', 'MinPlayer=2', 'SkipScore=no', 'TrainCrate=no', 'TruckCrate=no',
                           'AltHomeCell=99', 'OneTimeOnly=no', 'CarryOverCap=0', 'NewINIFormat=4',
                           'NextScenario=', 'CarryOverMoney=0.000000', 'AltNextScenario=', 'MultiplayerOnly=1',
                           'IceGrowthEnabled=yes', 'VeinGrowthEnabled=yes', 'TiberiumGrowthEnabled=yes',
                           'IgnoreGlobalAITriggers=no', 'TiberiumDeathToVisceroid=no']))
    secs.append(('Map', [f'Size=0,0,{W},{H}', 'Theater=TEMPERATE', f'LocalSize={LX},{LY},{LW},{LH}']))
    secs.append(('Lighting', ['Red=1.000000', 'Blue=0.950000', 'Green=0.980000', 'Level=0.032000',
                              'Ground=0.000000', 'IonRed=1.100000', 'Ambient=1.050000', 'IonBlue=1.200000',
                              'IonGreen=1.000000', 'IonLevel=0.010000', 'IonGround=0.000000',
                              'IonAmbient=0.380000', 'DominatorRed=0.850000', 'DominatorBlue=0.300000',
                              'DominatorGreen=0.200000', 'DominatorLevel=0.000000', 'DominatorGround=0.000000',
                              'DominatorAmbient=1.500000', 'DominatorAmbientChangeRate=0.009000']))
    secs.append(('SpecialFlags', ['Inert=no', 'FogOfWar=no', 'IonStorms=no', 'MCVDeploy=no', 'Meteorites=no',
                                  'Visceroids=no', 'FixedAlliance=no', 'TiberiumGrows=yes', 'InitialVeteran=no',
                                  'HarvesterImmune=no', 'TiberiumSpreads=yes', 'TiberiumExplosive=no',
                                  'DestroyableBridges=yes']))
    secs.append(('Houses', [f'{i}={c}' for i, c in enumerate(COUNTRY)]))
    for c, color in COUNTRY.items():
        secs.append((c, ['IQ=0', 'Edge=North', f'Color={color}', f'Allies={c}', f'Country={c}', 'Credits=0',
                         'NodeCount=0', 'TechLevel=1', 'PercentBuilt=0', 'PlayerControl=no']))
    secs.append(('Waypoints', [f'{i}={ry * 1000 + rx}' for i, (rx, ry) in enumerate(start_cells)]))
    secs.append(('Structures', [f'{i}={s}' for i, s in enumerate(structures)]))
    secs.append(('Terrain', [f'{k}={v}' for k, v in sorted(terrain.items())]))
    secs.append(('IsoMapPack5', [f'{i + 1}={l}' for i, l in enumerate(C.to_ini_lines(C.pack(bytes(iso), 5)))]))
    secs.append(('OverlayPack', [f'{i + 1}={l}' for i, l in enumerate(C.to_ini_lines(C.pack(bytes(ov), 80)))]))
    secs.append(('OverlayDataPack', [f'{i + 1}={l}' for i, l in enumerate(C.to_ini_lines(C.pack(bytes(od), 80)))]))
    for s in ('Triggers', 'Events', 'Actions', 'Tags', 'CellTags', 'TaskForces', 'TeamTypes', 'ScriptTypes'):
        secs.append((s, []))
    secs.append(('Digest', ['1=' + C.to_ini_lines(bytes(rng.integers(0, 256, 20, dtype=np.uint8)))[0]]))
    fix = {'PreviewPack'}
    out = ['; Colorado - Red Alert 2 multiplayer map generated from real elevation (AWS Terrain Tiles)',
           '; and Natural Earth rivers. 8 players, temperate.', '']
    for name, lines in secs:
        out.append(f'[{name}]')
        out += [f'{i + 1}={l}' for i, l in enumerate(lines)] if name in fix else lines
        out.append('')
    return '\r\n'.join(out) + '\r\n'


def report():
    print('cells', len(CELLS), 'levels', np.bincount(level[VALID]))
    print('ramps', int((VALID & (ramp > 0)).sum()), 'flat', int(flat.sum()))
    print('trees', len(terrain), 'ore cells', int(((overlay >= 100) & (overlay != 255)).sum()),
          'gem cells', int((overlay < 100).sum()), 'structures', len(structures))
    for (name, *_), (x, y) in zip(STARTS, start_cells):
        others = [math.hypot(x - a, y - b) for a, b in start_cells if (a, b) != (x, y)]
        ore = int(((overlay != 255) & (np.hypot(CX - x, CY - y) < 14)).sum())
        print(f'  {name:18s} cell=({x},{y}) level={level[x, y]} nearest={min(others):.0f} ore<14={ore}')


if __name__ == '__main__':
    ini = build_ini()
    open(os.path.join(OUT, 'colorado.mpr'), 'w', newline='').write(ini)
    Image.fromarray(prev).resize((4 * W, 2 * H), Image.NEAREST).save(os.path.join(OUT, 'preview_small.png'))
    np.savez(os.path.join(OUT, 'state.npz'), heights=heights, level=level, ramp=ramp, tile=tile,
             overlay=overlay, valid=VALID, cover=cover)
    json.dump({'terrain': terrain, 'structures': structures, 'starts': start_cells,
               'names': [s[0] for s in STARTS]}, open(os.path.join(OUT, 'objects.json'), 'w'))
    report()
