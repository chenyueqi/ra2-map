"""Generate a Red Alert 2 multiplayer map of a real place from elevation and river data.

usage: python3 gen.py <region>     (region module in regions/, e.g. colorado, hohhot)
"""
import glob
import importlib
import json
import math
import os
import struct
import sys

import numpy as np
from PIL import Image

import ra2codec as C
import tileset

HERE = os.path.dirname(os.path.abspath(__file__))
REGION = sys.argv[1] if len(sys.argv) > 1 else 'colorado'
cfg = importlib.import_module(f'regions.{REGION}')
g = sys.modules[__name__]           # engine context handed to the region's ground_cover()
OUT = os.path.join(HERE, 'out', REGION)
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(cfg.SEED)
THEATER = getattr(cfg, 'THEATER', 'TEMPERATE')
TS = tileset.load(os.path.join(HERE, 'ref', {'TEMPERATE': 'temperat.yaml', 'URBAN': 'urban.yaml'}[THEATER]))

# --- map geometry --------------------------------------------------------------
W, H = cfg.W, cfg.H
LX, LY, LW, LH = cfg.LX, cfg.LY, cfg.LW, cfg.LH
N = W + H + 4                      # corner grid is N x N, indexed [rx, ry]
LON0, LON1, LAT0, LAT1 = cfg.LON0, cfg.LON1, cfg.LAT0, cfg.LAT1
X0, X1 = -(W - 2 * LX), 2 * (LX + LW) - W        # rx - ry range of the visible area
Y0, Y1 = W + 2 * LY, W + 2 * (LY + LH + 1)       # rx + ry range of the visible area


def geo(X, Y):
    return LON0 + (X - X0) / (X1 - X0) * (LON1 - LON0), LAT0 + (Y - Y0) / (Y1 - Y0) * (LAT1 - LAT0)


def to_iso(lon, lat):
    """Continuous (rx, ry) of a lon/lat point (cell centers sit at integer + 0.5)."""
    X = X0 + (lon - LON0) / (LON1 - LON0) * (X1 - X0)
    Y = Y0 + (lat - LAT0) / (LAT1 - LAT0) * (Y1 - Y0)
    return (X + Y) / 2, (Y - X) / 2


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
Z = cfg.DEM_ZOOM


def load_dem():
    tiles = glob.glob(os.path.join(HERE, cfg.DEM_DIR, f'{Z}_*.png'))
    key = [tuple(int(v) for v in os.path.basename(t)[:-4].split('_')[1:]) for t in tiles]
    tx0, ty0 = min(k[0] for k in key), min(k[1] for k in key)
    E = np.zeros((256 * (max(k[1] for k in key) - ty0 + 1), 256 * (max(k[0] for k in key) - tx0 + 1)))
    for t, (x, y) in zip(tiles, key):
        a = np.asarray(Image.open(t).convert('RGB'), float)
        E[(y - ty0) * 256:(y - ty0 + 1) * 256, (x - tx0) * 256:(x - tx0 + 1) * 256] = \
            a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768
    return E, tx0, ty0


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


DEM, TX0, TY0 = load_dem()
DEM_SMOOTH = box_blur(DEM, cfg.BLUR_SMOOTH)
DEM_WIDE = box_blur(DEM, cfg.BLUR_WIDE)
DEM_TERRAIN = box_blur(DEM, cfg.BLUR_TERRAIN)
ELEV_CORNER = sample(DEM_TERRAIN, CORNER_LON, CORNER_LAT)
ELEV_CELL = sample(DEM_SMOOTH, CELL_LON, CELL_LAT)
RELIEF_CELL = np.abs(ELEV_CELL - sample(DEM_WIDE, CELL_LON, CELL_LAT))


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


heights = relax_min(np.round(np.interp(ELEV_CORNER, cfg.ELEV_PTS, cfg.LVL_PTS)))


def dist_to_point(lon, lat):
    rx, ry = to_iso(lon, lat)
    return np.hypot(CX + 0.5 - rx, CY + 0.5 - ry)


def polyline_distance(lines, step=0.25, offset=0.5):
    """Distance (in cells) from each cell center (offset .5) or corner (offset 0) to the nearest polyline."""
    pts = []
    for line in lines:
        for (a, b), (c, d) in zip(line, line[1:]):
            p, q = to_iso(a, b), to_iso(c, d)
            n = max(1, int(math.hypot(q[0] - p[0], q[1] - p[1]) / step))
            pts += [(p[0] + (q[0] - p[0]) * k / n, p[1] + (q[1] - p[1]) * k / n) for k in range(n + 1)]
    dist = np.full((N, N), np.inf)
    for px, py in pts:
        x0, y0 = int(px) - 4, int(py) - 4
        if x0 < -8 or y0 < -8 or x0 >= N or y0 >= N:
            continue
        xs, ys = slice(max(x0, 0), min(x0 + 9, N)), slice(max(y0, 0), min(y0 + 9, N))
        d = np.hypot(CX[xs, ys] + offset - px, CY[xs, ys] + offset - py)
        dist[xs, ys] = np.minimum(dist[xs, ys], d)
    return dist


# --- open water: block-level marching squares with the shore pieces learned from real maps ---------
WATER_2X2 = [314, 315, 316, 317, 318, 319, 320, 321]
SHORE_BY_CASE = {  # corner labels top,right,bottom,left of a 2x2 block -> shore templates (see learn_shore.py)
    'LLWW': [90, 91], 'LWWW': [95], 'LWWL': [97, 98, 99], 'WWWL': [103], 'WWLL': [106, 107],
    'WWLW': [111, 112], 'WLLW': [113, 115], 'WLWW': [119, 120], 'LLWL': [121, 122], 'LWLL': [124],
    'WLLL': [125, 126], 'LLLW': [127]}
fixed_tile = np.full((N, N), -1)
fixed_sub = np.zeros((N, N), int)
water_cell = np.zeros((N, N), bool)     # water or shore: no ore, trees, buildings, LAT


def components(mask):
    """4-connected components of a boolean grid: (label array, sizes)."""
    lab = np.full(mask.shape, -1)
    sizes = []
    for start in zip(*np.nonzero(mask)):
        if lab[start] >= 0:
            continue
        stack, n = [start], 0
        lab[start] = len(sizes)
        while stack:
            x, y = stack.pop()
            n += 1
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1] and mask[nx, ny] and lab[nx, ny] < 0:
                    lab[nx, ny] = len(sizes)
                    stack.append((nx, ny))
        sizes.append(n)
    return lab, np.array(sizes)


def build_water():
    water_corners = getattr(cfg, 'water_corners', None)
    if not cfg.WATER_RIVERS and not cfg.LAKES and not water_corners:
        return
    wd = polyline_distance(cfg.WATER_RIVERS, offset=0.0) if cfg.WATER_RIVERS else np.full((N, N), np.inf)
    wet = wd < cfg.WATER_HALF_WIDTH
    for lon, lat, a_km, b_km in cfg.LAKES:   # ellipse in km (east-west, north-south)
        dx = (CORNER_LON - lon) * 111.32 * math.cos(math.radians(lat)) / a_km
        dy = (CORNER_LAT - lat) * 111.0 / b_km
        wet |= dx * dx + dy * dy < 1
    if water_corners:
        wet |= water_corners(g)
    keep_land = np.zeros((N, N), bool)
    for lon, lat, r in getattr(cfg, 'ISLANDS', []):    # islands too small for the DEM at this scale
        rx, ry = to_iso(lon, lat)
        keep_land |= np.hypot(CX - rx, CY - ry) < r
    wet &= ~keep_land
    B = N // 2
    lab = wet[0:2 * B:2, 0:2 * B:2].copy()     # labels on block corners (even corner coordinates)
    keep = keep_land[0:2 * B:2, 0:2 * B:2]
    for kind, min_size in ((True, getattr(cfg, 'MIN_WATER_CORNERS', 0)), (False, getattr(cfg, 'MIN_LAND_CORNERS', 0))):
        comp, sizes = components(lab == kind)
        small = (comp >= 0) & (sizes[comp] < min_size)
        if not kind:
            small &= ~np.isin(comp, np.unique(comp[keep & (comp >= 0)]))
        lab[small] = not kind
    while True:                                # remove saddles (opposite corners wet) by flooding them
        t, r, b, l = lab[:-1, :-1], lab[1:, :-1], lab[1:, 1:], lab[:-1, 1:]
        saddle = (t & b & ~r & ~l) | (r & l & ~t & ~b)
        if not saddle.any():
            break
        for i, j in zip(*np.nonzero(saddle)):
            lab[i:i + 2, j:j + 2] = True
    for i in range(B - 1):
        for j in range(B - 1):
            case = ''.join('W' if v else 'L' for v in (lab[i, j], lab[i + 1, j], lab[i + 1, j + 1], lab[i, j + 1]))
            if case == 'LLLL':
                continue
            tmpl = WATER_2X2 if case == 'WWWW' else SHORE_BY_CASE[case]
            t = tmpl[int(rng.integers(len(tmpl)))]
            for oy in range(2):
                for ox in range(2):
                    x, y = 2 * i + ox, 2 * j + oy
                    fixed_tile[x, y], fixed_sub[x, y] = t, oy * 2 + ox
                    water_cell[x, y] = True


build_water()

# --- flatten building ground ------------------------------------------------------
lock = np.full((N, N), np.nan)
if water_cell.any():
    wet_corner = np.zeros((N, N), bool)
    for dx in (0, 1):
        for dy in (0, 1):
            wet_corner |= shift(water_cell, -dx, -dy, False)
    near = wet_corner | shift(wet_corner, 1, 0, False) | shift(wet_corner, -1, 0, False) | \
        shift(wet_corner, 0, 1, False) | shift(wet_corner, 0, -1, False)
    lock[near] = cfg.WATER_LEVEL


def lock_disc(lon, lat, r):
    rx, ry = to_iso(lon, lat)
    d = np.hypot(CX - rx, CY - ry)
    area = d <= r + 0.75
    lvl = float(np.round(np.median(heights[d <= max(r * 0.6, 2)])))
    near = lock[d <= r + 5]
    if (~np.isnan(near)).any():         # join a neighbouring flat area instead of stepping next to it
        lvl = float(np.round(np.nanmedian(near)))
    if (~np.isnan(lock)).any():         # stay reachable by 1-level ramps from every flat area already placed
        hi = relax_min(np.where(np.isnan(lock), np.inf, lock))[area].min()
        lo = -relax_min(np.where(np.isnan(lock), np.inf, -lock))[area].max()
        lvl = float(min(max(lvl, lo), hi))
    lock[area & np.isnan(lock)] = lvl
    return lvl


for _, lon, lat in cfg.STARTS:
    lock_disc(lon, lat, cfg.START_FLAT)
for _, lon, lat in cfg.ORE_TOWNS:
    lock_disc(lon, lat, 4.5)
for _, lon, lat in cfg.GEM_TOWNS:
    lock_disc(lon, lat, 3.5)
for _, lon, lat in cfg.OIL:
    lock_disc(lon, lat, 3)
for _, _, lon, lat, w, h in cfg.TECH:
    lock_disc(lon, lat, max(w, h) + 1.5)
for _, _, lon, lat, w, h in getattr(cfg, 'LANDMARKS', []):
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
assert not (water_cell & VALID & (ramp > 0)).any(), 'water must be flat'
flat = VALID & (ramp == 0)

# --- noise --------------------------------------------------------------------------


def value_noise(scale, seed):
    a = np.random.default_rng(seed).random((N // scale + 3, N // scale + 3))
    img = Image.fromarray((a * 255).astype(np.uint8)).resize(((N // scale + 3) * scale,) * 2, Image.BICUBIC)
    return np.asarray(img, float)[:N, :N] / 255


NOISE = 0.6 * value_noise(6, 1) + 0.4 * value_noise(3, 2)
NOISE2 = value_noise(4, 3)

# --- rivers and roads ---------------------------------------------------------------------
river_lines = list(cfg.EXTRA_RIVERS)
x0b, y0b, x1b, y1b = cfg.RIVER_BBOX
for f in ('data/rivers.geojson', 'data/rivers_na.geojson'):
    for ft in json.load(open(os.path.join(HERE, f)))['features']:
        geom = ft['geometry']
        if not geom or ft['properties'].get('name') not in cfg.RIVER_NAMES:
            continue
        for line in (geom['coordinates'] if geom['type'] == 'MultiLineString' else [geom['coordinates']]):
            if any(x0b <= x <= x1b and y0b <= y <= y1b for x, y in line):
                river_lines.append(line)
RIVER_D = polyline_distance(river_lines) if river_lines else np.full((N, N), np.inf)
ROAD_D = polyline_distance(cfg.ROADS.values())

# --- ground cover -------------------------------------------------------------------
GRASS, THICK, ROUGH, SAND, PAVE = 0, 131, 418, 493, 534
LAT_START = {THICK: 132, ROUGH: 419, SAND: 494, PAVE: 463}

cover, mountain, tree_p = cfg.ground_cover(g)
city = np.zeros((N, N), bool)
for _, lon, lat, r in cfg.CITIES:
    city |= dist_to_point(lon, lat) < r + 1.6 * (NOISE2 - 0.5)
if hasattr(cfg, 'urban_mask'):           # whole built-up area, not just the city centres
    city |= cfg.urban_mask(g) & VALID & ~water_cell
cover[city] = getattr(cfg, 'CITY_COVER', PAVE)
if hasattr(cfg, 'city_cover'):          # e.g. plazas inside the city
    cfg.city_cover(g, cover, city)
cover[~flat | water_cell] = GRASS

tile = np.zeros((N, N), int)
sub = np.zeros((N, N), int)
for rx, ry in CELLS:
    if water_cell[rx, ry]:
        tile[rx, ry], sub[rx, ry] = fixed_tile[rx, ry], fixed_sub[rx, ry]
        continue
    if not flat[rx, ry]:
        tile[rx, ry] = RAMP_TILE0 + ramp[rx, ry]
        continue
    s = cover[rx, ry]
    if s == GRASS:
        continue
    mask = 0
    for bit, (dx, dy) in enumerate(((0, -1), (1, 0), (0, 1), (-1, 0))):
        nx, ny = rx + dx, ry + dy
        same = not (0 <= nx < N and 0 <= ny < N) or not VALID[nx, ny] or \
            (flat[nx, ny] and not water_cell[nx, ny] and cover[nx, ny] == s)
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
    m = blob(cx, cy, r) & flat & ~taken & VALID & ~water_cell
    if kind == 'ore':
        overlay[m] = np.where(rng.random(m.sum()) < 0.85, ORE, ORE_B)
    else:
        overlay[m] = np.where(rng.random(m.sum()) < 0.85, GEM, GEM_B)
    return int(m.sum())


start_cells = []
center = to_iso(*cfg.MAP_CENTER)
for name, lon, lat in cfg.STARTS:
    rx, ry = to_iso(lon, lat)
    start_cells.append((int(rx), int(ry)))
    taken |= np.hypot(CX + 0.5 - rx, CY + 0.5 - ry) < 5.5
for (name, lon, lat), (sx, sy) in zip(cfg.STARTS, start_cells):
    clear = getattr(cfg, 'START_WATER_CLEAR', 6)
    assert not water_cell[sx - clear:sx + clear + 1, sy - clear:sy + clear + 1].any(), f'{name} start is too close to water'
    away = math.atan2(sy + 0.5 - center[1], sx + 0.5 - center[0])
    placed = 0
    for dist in (cfg.START_ORE_DIST, cfg.START_ORE_DIST - 1.5, cfg.START_ORE_DIST + 1.5):
        for da in (0.9, -0.9, 0.3, -0.3, 1.6, -1.6, 2.4, -2.4):
            if placed == 2:
                break
            a = away + da
            fx, fy = sx + 0.5 + dist * math.cos(a), sy + 0.5 + dist * math.sin(a)
            if not in_local(fx, fy, 6):
                continue
            if (overlay[int(fx) - 3:int(fx) + 4, int(fy) - 3:int(fy) + 4] != 255).any():
                continue
            n = place_field(fx, fy, 2.9, 'ore')
            if n >= 18:
                placed += 1
    assert placed >= getattr(cfg, 'START_FIELDS_MIN', 2), name
for _, lon, lat in cfg.ORE_TOWNS:
    place_field(*to_iso(lon, lat), 2.6, 'ore')
for _, lon, lat in cfg.GEM_TOWNS:
    place_field(*to_iso(lon, lat), 2.0, 'gem')

# --- structures (neutral tech) ----------------------------------------------------------
structures = []


def place_building(kind, lon, lat, w, h):
    rx, ry = to_iso(lon, lat)
    x0, y0 = int(rx - w / 2), int(ry - h / 2)

    def ok(x, y):
        foot = (slice(x, x + w), slice(y, y + h))
        return (flat[foot].all() and VALID[foot].all() and not water_cell[foot].any() and not taken[foot].any()
                and len(set(level[foot].ravel())) == 1)
    # the real spot can sit right on a shoreline at this scale: take the nearest dry flat footprint
    spots = sorted(((dx, dy) for dx in range(-5, 6) for dy in range(-5, 6)), key=lambda d: math.hypot(*d))
    x, y = next(((x0 + dx, y0 + dy) for dx, dy in spots if ok(x0 + dx, y0 + dy)), (None, None))
    assert x is not None, (kind, lon, lat)
    taken[x - 1:x + w + 1, y - 1:y + h + 1] = True
    overlay[x - 1:x + w + 1, y - 1:y + h + 1] = 255
    structures.append(f'Neutral,{kind},256,{x},{y},64,None,1,0,1,0,0,None,None,None,0,0')


for _, lon, lat in cfg.OIL:
    place_building('CAOILD', lon, lat, 2, 2)
for kind, _, lon, lat, w, h in cfg.TECH:
    place_building(kind, lon, lat, w, h)
for kind, _, lon, lat, w, h in getattr(cfg, 'LANDMARKS', []):
    place_building(kind, lon, lat, w, h)

# --- civilian buildings: city blocks with streets between them -------------------------------
props = {}
GAP = {'farm': 4, 'suburb': 2, 'campus': 2}         # street width around a building (cells)


def place_props(placed, pool):
    """A few street lights / traffic lights on open corners next to buildings (never in a one-cell lane)."""
    for x, y, w, h in placed:
        if pool == 'farm' or rng.random() >= cfg.PROP_DENSITY:
            continue
        px, py = x + w, y - 1
        if not (VALID[px, py] and flat[px, py] and in_local(px, py, 3)) or taken[px, py] or water_cell[px, py]:
            continue
        open4 = sum(VALID[px + a, py + b] and flat[px + a, py + b] and not taken[px + a, py + b]
                    for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        if open4 >= 3 and overlay[px, py] == 255:
            kind = 'TRFF0' if pool == 'highrise' and rng.random() < 0.4 else 'LT_GEN0'
            props[py * 1000 + px] = f'{kind}{int(rng.integers(1, 5))}'
            taken[px, py] = True


def place_civilians(cand, pool, density, inside=None):
    """Greedy placement over candidate cells in random order; returns the footprints placed."""
    gap = GAP.get(pool, 1)
    kinds = [k for k in cfg.CIVILIAN_POOLS[pool] if CIV[k]['available'] and max(CIV[k]['w'], CIV[k]['h']) <= 4]
    placed = []
    for i in rng.permutation(len(cand)):
        x, y = (int(v) for v in cand[i])
        if rng.random() >= density:
            continue
        for k in rng.permutation(kinds)[:4]:
            w, h = CIV[k]['w'], CIV[k]['h']
            if (inside is None or inside[x:x + w, y:y + h].all()) and civ_ok(x, y, w, h):
                blocked[x - gap:x + w + gap, y - gap:y + h + gap] = True
                taken[x:x + w, y:y + h] = True
                structures.append(f'Neutral,{k},256,{x},{y},64,None,1,0,1,0,0,None,None,None,0,0')
                placed.append((x, y, w, h))
                break
    place_props(placed, pool)
    return placed


def civ_ok(x, y, w, h):
    foot = (slice(x, x + w), slice(y, y + h))
    return (x > 0 and y > 0 and VALID[foot].all() and not blocked[foot].any()
            and len(set(level[foot].ravel())) == 1 and in_local(x, y, 3) and in_local(x + w, y + h, 3))


if getattr(cfg, 'CIVILIAN_ZONES', None):
    import buildings
    CIV = buildings.load(THEATER)
    keep_clear = np.zeros((N, N), bool)
    for sx, sy in (to_iso(lon, lat) for _, lon, lat in cfg.STARTS):
        keep_clear |= np.hypot(CX + 0.5 - sx, CY + 0.5 - sy) < cfg.CIVILIAN_START_CLEAR
    ore_near = overlay != 255
    for _ in range(2):
        ore_near = ore_near | shift(ore_near, 1, 0, False) | shift(ore_near, -1, 0, False) | \
            shift(ore_near, 0, 1, False) | shift(ore_near, 0, -1, False)
    blocked = taken | keep_clear | ore_near | water_cell | ~flat
    for lon, lat, r, pool, density in cfg.CIVILIAN_ZONES:
        zx, zy = to_iso(lon, lat)
        cand = [(x, y) for x in range(int(zx - r) - 2, int(zx + r) + 3) for y in range(int(zy - r) - 2, int(zy + r) + 3)
                if math.hypot(x + 0.5 - zx, y + 0.5 - zy) < r]
        place_civilians(cand, pool, density)
    if getattr(cfg, 'CIVILIAN_FILL', None):       # (pool, density) over the rest of the built-up area
        place_civilians(list(zip(*np.nonzero(city & flat))), *cfg.CIVILIAN_FILL, inside=city)

# ore density: fuller in the middle of a field, as FinalAlert does it
odata = np.zeros((N, N), int)
has = overlay != 255
cnt = sum(shift(has.astype(int), dx, dy, 0) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy)
odata[has] = np.clip(cnt[has] + 3, 3, 11)

# --- trees ------------------------------------------------------------------------------
p = tree_p.copy()
p[ROAD_D < 1.2] = 0
p[city & flat & ~water_cell] = 0.0
p[~np.isnan(lock)] *= 0.15
for sx, sy in start_cells:
    p[np.hypot(CX - sx, CY - sy) < cfg.START_FLAT + 1] = 0
p[taken | (overlay != 255) | (shift((overlay != 255).astype(int), 1, 0, 0) > 0) |
  (shift((overlay != 255).astype(int), 0, 1, 0) > 0) | water_cell] = 0
terrain = {}
for rx, ry in CELLS:
    if in_local(rx, ry, 2) and rng.random() < p[rx, ry]:
        trees = cfg.MOUNTAIN_TREES if mountain[rx, ry] else cfg.VALLEY_TREES
        terrain[ry * 1000 + rx] = trees[int(rng.integers(len(trees)))]
terrain.update(props)
for name, lon, lat in cfg.ORE_DRILLS:     # ore drills next to the mining camps
    rx, ry = to_iso(lon, lat)
    terrain[(int(ry) - 3) * 1000 + int(rx) + 3] = 'TIBTRE01'

# --- preview ---------------------------------------------------------------------------


def hexrgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)


def cell_color(rx, ry):
    col = hexrgb(TS[int(tile[rx, ry])]['tiles'][int(sub[rx, ry])]['color'])
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
        iso += struct.pack('<HHiBBB', rx, ry, int(tile[rx, ry]), int(sub[rx, ry]), int(level[rx, ry]), 0)
    ov = bytearray([255]) * (512 * 512)
    od = bytearray(512 * 512)
    for rx, ry in CELLS:
        if overlay[rx, ry] != 255:
            ov[rx + 512 * ry] = int(overlay[rx, ry])
            od[rx + 512 * ry] = int(odata[rx, ry])
    li = dict(cfg.LIGHTING)
    secs = []
    secs.append(('Header', header()))
    secs.append(('Preview', [f'Size=0,0,{2 * W},{H}']))
    secs.append(('PreviewPack', C.to_ini_lines(C.pack(prev.tobytes(), 5))))
    secs.append(('Basic', [f'Name={cfg.NAME}', 'Author=Yueqi Chen', 'Percent=0', 'GameMode=standard',
                           'HomeCell=98', 'InitTime=10000', 'Official=no', 'EndOfGame=no', 'FreeRadar=no',
                           f'MaxPlayer={len(cfg.STARTS)}', 'MinPlayer=2', 'SkipScore=no', 'TrainCrate=no',
                           'TruckCrate=no', 'AltHomeCell=99', 'OneTimeOnly=no', 'CarryOverCap=0',
                           'NewINIFormat=4', 'NextScenario=', 'CarryOverMoney=0.000000', 'AltNextScenario=',
                           'MultiplayerOnly=1', 'IceGrowthEnabled=yes', 'VeinGrowthEnabled=yes',
                           'TiberiumGrowthEnabled=yes', 'IgnoreGlobalAITriggers=no', 'TiberiumDeathToVisceroid=no']))
    secs.append(('Map', [f'Size=0,0,{W},{H}', f'Theater={THEATER}', f'LocalSize={LX},{LY},{LW},{LH}']))
    secs.append(('Lighting', [f'Red={li["Red"]:.6f}', f'Blue={li["Blue"]:.6f}', f'Green={li["Green"]:.6f}',
                              f'Level={li["Level"]:.6f}', 'Ground=0.000000', 'IonRed=1.100000',
                              f'Ambient={li["Ambient"]:.6f}', 'IonBlue=1.200000',
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
    out = [f'; {cfg.NAME} - Red Alert 2 multiplayer map generated from real elevation (AWS Terrain Tiles)',
           f'; and Natural Earth rivers. {len(cfg.STARTS)} players, {THEATER.lower()}.', '']
    for name, lines in secs:
        out.append(f'[{name}]')
        out += [f'{i + 1}={l}' for i, l in enumerate(lines)] if name == 'PreviewPack' else lines
        out.append('')
    return '\r\n'.join(out) + '\r\n'


def report():
    print('cells', len(CELLS), 'levels', np.bincount(level[VALID]))
    print('ramps', int((VALID & (ramp > 0)).sum()), 'flat', int(flat.sum()), 'water+shore', int((water_cell & VALID).sum()))
    print('trees', len(terrain), 'ore cells', int(((overlay >= 100) & (overlay != 255)).sum()),
          'gem cells', int((overlay < 100).sum()), 'structures', len(structures))
    for (name, *_), (x, y) in zip(cfg.STARTS, start_cells):
        others = [math.hypot(x - a, y - b) for a, b in start_cells if (a, b) != (x, y)]
        ore = int(((overlay != 255) & (np.hypot(CX - x, CY - y) < 14)).sum())
        print(f'  {name:18s} cell=({x},{y}) level={level[x, y]} nearest={min(others):.0f} ore<14={ore}')


if __name__ == '__main__':
    ini = build_ini()
    open(os.path.join(OUT, cfg.OUTFILE), 'w', newline='').write(ini)
    Image.fromarray(prev).resize((4 * W, 2 * H), Image.NEAREST).save(os.path.join(OUT, 'preview_small.png'))
    np.savez(os.path.join(OUT, 'state.npz'), heights=heights, level=level, ramp=ramp, tile=tile, sub=sub,
             overlay=overlay, valid=VALID, cover=cover)
    json.dump({'terrain': terrain, 'structures': structures, 'starts': start_cells,
               'names': [s[0] for s in cfg.STARTS], 'W': W, 'H': H, 'local': [LX, LY, LW, LH],
               'theater': THEATER},
              open(os.path.join(OUT, 'objects.json'), 'w'))
    report()
