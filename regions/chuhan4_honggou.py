"""楚汉争霸 4: 成皋之战 / 荥阳对峙 / 鸿沟议和 (204-203 BC). For two years 刘邦 and 项羽 fought over 荥阳, 成皋
(虎牢关) and the 敖仓 granary on the 敖山 above the 黄河. They faced each other across the 广武涧 ravine on the
广武山, 汉王城 on its west side and 霸王城 on its east, where 项羽 threatened to boil 刘太公. In autumn 203 BC they
agreed to split the world along the 鸿沟 canal: "中分天下，割鸿沟以西者为汉，以东者为楚" - the 楚河汉界 of
Chinese chess.

The map's dividing line is one unbroken stretch of water: the 广武涧 (drawn as the canal's head, from the 黄河
down through the 广武山) and the 鸿沟 itself, east past 荥阳 and 官渡, then south past 大梁 and 陈留. 汉 holds
the south and west, 楚 the north and east. Three crossings: 荥阳东, 官渡 and 陈留南.

The 黄河 runs along the north edge on its course of the time: as today below 巩 and 成皋, but from the 广武山
north-east toward 白马津 (the modern channel east of 广武 is much younger). One ford, 玉门津 north of 成皋 (刘邦
fled across it to 修武 in 204 BC), leads to the 河内 plain on the north bank.

West of 荥阳 the 嵩山 / 浮戏山 / 邙山 hills are a plateau of cliffs: the only way from 巩 (and 洛阳 beyond) to the
east is the 虎牢 defile between the cliffs and the river.

Town sites (古荥 for 荥阳, 汜水 / 虎牢关, 京襄城, 汉王城, 霸王城) come from OpenStreetMap (data/osm_chuhan4_honggou);
the 鸿沟 and the old 黄河 below 广武 are traced by hand."""
import os

import numpy as np

from regions import osmwater

NAME = 'Chu-Han 4: Chenggao and the Hong Canal'
OUTFILE = 'chuhan4_honggou.mpr'
SEED = 203  # the 鸿沟 treaty, autumn 203 BC
W, H = 200, 134
LX, LY, LW, LH = 2, 4, 196, 121
LON0, LON1, LAT0, LAT1 = 112.95, 114.80, 35.25, 34.30418
DEM_DIR, DEM_ZOOM = 'data/dem_chuhan4_honggou', 11
BLUR_SMOOTH, BLUR_WIDE, BLUR_TERRAIN = 2, 10, 3
ELEV_PTS = [0, 92, 108, 150, 200, 300, 360]       # 黄河 bottomland and the eastern plain at 0, 荥阳 at 1, 广武山 at 4-5
LVL_PTS = [0, 0, 1, 2, 3, 4, 5]
# The 广武山 is a loess ridge only 3 km wide and ~100 m above the 荥阳 plain, squeezed between the 黄河 and the
# canal: at 0.6 km a cell, and with the ground next to water held at the water's level, it would not show. The DEM
# is prepared with a dome added over it (a stylised, somewhat wider ridge) and the 广武涧 notch left at its real
# floor, so the two camps face each other across a ravine. Raw tiles: data/dem_chuhan4_honggou/raw.
GUANGWU = (113.455, 34.944, 8.5, 3.4, 140)        # lon, lat, half-length E-W (km), half-width N-S (km), height (m)
RAVINE_LON, RAVINE_HALF_KM = 113.462, 0.35


def _prepare_dem():
    """data/dem_chuhan4_honggou/raw/*.png (AWS terrarium tiles) -> data/dem_chuhan4_honggou/*.png, 广武山 raised."""
    import glob
    import math
    from PIL import Image
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', DEM_DIR)
    tiles = sorted(glob.glob(os.path.join(here, 'raw', f'{DEM_ZOOM}_*.png')))
    stamp = repr((GUANGWU, RAVINE_LON, RAVINE_HALF_KM))
    stamp_file = os.path.join(here, 'prepared.txt')
    if not tiles or (os.path.exists(stamp_file) and open(stamp_file).read() == stamp):
        return
    key = [tuple(int(v) for v in os.path.basename(t)[:-4].split('_')[1:]) for t in tiles]
    tx0, ty0 = min(k[0] for k in key), min(k[1] for k in key)
    E = np.zeros((256 * (max(k[1] for k in key) - ty0 + 1), 256 * (max(k[0] for k in key) - tx0 + 1)))
    for t, (x, y) in zip(tiles, key):
        a = np.asarray(Image.open(t).convert('RGB'), float)
        E[(y - ty0) * 256:(y - ty0 + 1) * 256, (x - tx0) * 256:(x - tx0 + 1) * 256] = \
            a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768
    n = 256 * 2 ** DEM_ZOOM
    lon = (tx0 * 256 + np.arange(E.shape[1]) + 0.5) / n * 360 - 180
    lat = np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * (ty0 * 256 + np.arange(E.shape[0]) + 0.5) / n))))
    LON, LAT = np.meshgrid(lon, lat)
    hl, ha, a_km, b_km, top = GUANGWU
    kx = 111.32 * math.cos(math.radians(ha))
    d = np.hypot((LON - hl) * kx / a_km, (LAT - ha) * 111.32 / b_km)
    dome = top * 0.5 * (1 + np.cos(np.pi * np.clip(d, 0, 1)))          # smooth dome, 0 at the rim
    notch = np.clip((np.abs(LON - RAVINE_LON) * kx - RAVINE_HALF_KM) / 0.5, 0, 1)
    E = E + dome * notch
    v = np.round((E + 32768) * 256).astype(np.int64)
    rgb = np.stack([v >> 16, (v >> 8) & 255, v & 255], axis=-1).astype(np.uint8)
    for t, (x, y) in zip(tiles, key):
        Image.fromarray(rgb[(y - ty0) * 256:(y - ty0 + 1) * 256, (x - tx0) * 256:(x - tx0 + 1) * 256]).save(
            os.path.join(here, os.path.basename(t)))
    open(stamp_file, 'w').write(stamp)


_prepare_dem()
CLIFF_FOOT = 2                                    # every cliff stands on level 2; its top is level 6
PLATEAU_ELEV = 265                                # the hills above this contour (m) are cliff plateau ...
PLATEAU_ELEV_HULAO = (113.27, 34.775, 180)        # ... and above 180 m west of 113.27 E, north of 34.775 N (邙山 at 虎牢)
PLATEAU_ELEV_PTS = [265, 400, 650, 950, 1300]
PLATEAU_LVL_PTS = [6, 8, 10, 12, 13]              # 嵩山 up to level 13
PLATEAU_TREES = 0.05
LIGHTING = dict(Red=1.03, Green=1.0, Blue=0.94, Level=0.03, Ambient=1.02)   # autumn, dust over the loess

PASSES = {  # the 虎牢 defile: 巩 down the 洛水 to the 黄河, along the south bank to 汜水 / 成皋
    '虎牢道': [(113.016, 34.749), (113.03, 34.775), (113.045, 34.80), (113.065, 34.818), (113.10, 34.828),
              (113.14, 34.835), (113.175, 34.840), (113.198, 34.848), (113.215, 34.845), (113.25, 34.838)],
}
PASS_HALF_WIDTH = 3.0
RIDGES = {}
RIDGE_HALF_WIDTH = 3.0

STARTS = [  # 汉 south and west of the 鸿沟, 楚 north and east
    ('成皋', 113.285, 34.832),         # 汜水 / 虎牢关, taken and retaken three times
    ('汉王城', 113.41, 34.862),        # 刘邦's camp west of the 广武涧 (the start sits at the foot of the 广武山)
    ('巩', 113.016, 34.752),           # the road west to 洛阳, behind the 虎牢 defile
    ('京', 113.444, 34.716),           # 京索之战: 汉 cavalry stopped 楚 here in 205 BC
    ('霸王城', 113.60, 34.957),        # 项羽's camp east of the 广武涧 (start east of the real site)
    ('阳武', 113.94, 35.035),
    ('大梁', 114.34, 34.825),          # the old 魏 capital on the 鸿沟
    ('陈留', 114.62, 34.655),
]
START_FLAT, START_ORE_DIST, START_WATER_CLEAR = 8, 7.5, 3
MAP_CENTER = (113.80, 34.80)
ORE_TOWNS = [  # 汉 side
             ('荥阳', 113.534, 34.872), ('索', 113.37, 34.79), ('上街', 113.33, 34.775), ('管城', 113.66, 34.75),
             ('圃田', 113.80, 34.77), ('中牟', 113.98, 34.715), ('苑陵', 113.86, 34.53), ('新郑', 113.735, 34.40),
             ('华阳', 113.62, 34.55), ('尉氏', 114.19, 34.41), ('通许', 114.45, 34.48), ('启封', 114.20, 34.57),
             ('须水', 113.52, 34.80), ('京南', 113.50, 34.62),
             # 河内, north bank
             ('温', 113.08, 34.94), ('武陟', 113.40, 35.10), ('怀', 113.22, 35.05), ('州', 113.0, 35.05),
             ('山阳', 113.62, 35.18),
             # 楚 side
             ('广武东', 113.70, 34.97), ('原武', 113.78, 35.02), ('酸枣', 114.08, 35.10), ('封丘', 114.42, 35.04),
             ('平丘', 114.62, 35.00), ('小黄', 114.55, 34.88), ('浚仪北', 114.20, 34.92), ('外黄', 114.75, 34.75),
             ('雍丘', 114.75, 34.48), ('圉', 114.62, 34.38), ('阳武东', 114.10, 34.96)]
GEM_TOWNS = [('敖仓', 113.355, 34.922),      # the granary on the 敖山 that fed 汉 for two years
             ('修武道', 113.30, 35.13),      # 河内: where 刘邦 took over 韩信's army
             ('新郑 (韩故都)', 113.69, 34.44),
             ('外黄 (彭越)', 114.73, 34.585)]   # 彭越 raided 楚 supply lines here
OIL = []
TECH = [('CAOUTP', '虎牢关', 113.19, 34.84, 4, 3)]
LANDMARKS = [('CATS01', '敖仓', 113.33, 34.932, 2, 2)]         # twin silos for the granary
FLAT_CITIES = []
CITIES = []                                       # walled towns of earth, no paved streets
BRIDGES = [  # crossings of the 鸿沟 and the 玉门津 ford over the 黄河
    ('玉门津', 113.226, 34.90),
    ('荥阳东', 113.74, 34.858),
    ('官渡', 114.08, 34.737),
    ('陈留南', 114.515, 34.45),
]
HONGGOU = [  # the 广武涧 down from the 黄河 through the 广武山, then the canal (蒗荡渠) east and south
    (113.462, 34.985), (113.462, 34.955), (113.463, 34.93), (113.50, 34.912), (113.56, 34.902), (113.62, 34.885),
    (113.70, 34.862), (113.80, 34.832), (113.90, 34.80), (114.00, 34.765), (114.08, 34.737), (114.16, 34.728),
    (114.22, 34.718), (114.30, 34.68), (114.40, 34.643), (114.49, 34.615), (114.515, 34.50), (114.53, 34.40),
    (114.545, 34.29), (114.56, 34.16)]
YELLOW_RIVER = [  # as today down to the 广武山 (OSM), then north-east on its Western Han course toward 白马津
    (112.90, 34.8503), (112.921, 34.8424), (112.9404, 34.84), (112.9599, 34.8459), (112.973, 34.8561),
    (112.9855, 34.8656), (113.0048, 34.8603), (113.0204, 34.8513), (113.0394, 34.8469), (113.0571, 34.8421),
    (113.0731, 34.8379), (113.0918, 34.8406), (113.108, 34.847), (113.1273, 34.8481), (113.1469, 34.85),
    (113.1614, 34.8444), (113.1781, 34.8496), (113.1916, 34.8563), (113.2044, 34.8656), (113.2157, 34.8768),
    (113.2228, 34.8914), (113.2287, 34.9055), (113.2348, 34.9206), (113.2481, 34.9278), (113.262, 34.937),
    (113.2718, 34.95), (113.2864, 34.9558), (113.3079, 34.9528), (113.3262, 34.9484), (113.3401, 34.9545),
    (113.3511, 34.9674), (113.3651, 34.9748), (113.3782, 34.9832), (113.3938, 34.9892), (113.4114, 34.9919),
    (113.4362, 34.985), (113.462, 34.987), (113.50, 34.998), (113.58, 35.02), (113.68, 35.05), (113.78, 35.085),
    (113.86, 35.125), (113.93, 35.17), (113.99, 35.22), (114.03, 35.27), (114.07, 35.33)]
YELLOW_HALF_WIDTH = 1.8
TRACKS = {  # the roads of the time, dirt
    '三川东道 (洛阳-虎牢-荥阳)': [(112.95, 34.758), (113.016, 34.752), (113.04, 34.79), (113.065, 34.818),
                            (113.10, 34.828), (113.175, 34.840), (113.215, 34.845), (113.30, 34.84),
                            (113.385, 34.86), (113.45, 34.868), (113.534, 34.872)],
    '荥阳-官渡-大梁': [(113.534, 34.872), (113.62, 34.84), (113.70, 34.82), (113.80, 34.77), (113.98, 34.715),
                   (114.08, 34.737), (114.18, 34.77), (114.34, 34.825), (114.55, 34.88), (114.80, 34.90)],
    '荥阳-阳武': [(113.534, 34.872), (113.65, 34.87), (113.74, 34.858), (113.80, 34.92), (113.94, 35.035),
               (114.08, 35.10), (114.25, 35.20)],
    '大梁-陈留-雍丘': [(114.34, 34.825), (114.42, 34.75), (114.528, 34.67), (114.62, 34.655), (114.75, 34.48),
                   (114.80, 34.44)],
    '陈留南道': [(114.528, 34.67), (114.515, 34.45), (114.45, 34.40), (114.19, 34.41)],
    '京-索-荥阳': [(113.444, 34.716), (113.40, 34.76), (113.37, 34.79), (113.40, 34.83), (113.534, 34.872)],
    '新郑-京': [(113.735, 34.40), (113.62, 34.55), (113.52, 34.65), (113.444, 34.716)],
    '新郑-中牟': [(113.735, 34.40), (113.86, 34.53), (113.98, 34.715)],
    '甬道 (敖仓-荥阳)': [(113.33, 34.932), (113.36, 34.905), (113.40, 34.885), (113.45, 34.88), (113.534, 34.872)],
    '霸王城-阳武': [(113.60, 34.957), (113.78, 35.02), (113.94, 35.035)],
    '河内道': [(113.226, 34.915), (113.22, 34.96), (113.22, 35.05), (113.40, 35.10), (113.62, 35.18)],
    '温-州': [(113.22, 34.96), (113.08, 34.94), (113.0, 35.05)],
}
TRACK_COVER = 'SAND'
ROADS = dict(TRACKS)
RIVER_NAMES = set()
RIVER_BBOX = (112.8, 34.2, 114.9, 35.4)
EXTRA_RIVERS = osmwater.lines('data/osm_chuhan4_honggou/water.json', {'洛河', '索河', '须水河', '双洎河'}) + [
    [(113.20, 34.70), (113.205, 34.76), (113.21, 34.80), (113.212, 34.845), (113.216, 34.875)],   # 汜水 (by hand)
]
WATER_RIVERS = [HONGGOU]
WATER_HALF_WIDTH = 0.9
LAKES = []
WATER_LEVEL = 0
MIN_WATER_CORNERS, MIN_LAND_CORNERS = 6, 8
MOUNTAIN_TREES = ['TREE25', 'TREE26', 'TREE27', 'TREE05', 'TREE06', 'TREE07', 'TREE08', 'TREE15']
VALLEY_TREES = ['TREE01', 'TREE02', 'TREE03', 'TREE04', 'TREE10', 'TREE11', 'TREE12', 'TREE17']
CHECK_POINTS = [('敖仓', 113.355, 34.922), ('河内', 113.30, 35.13), ('汉王城址', 113.4546, 34.9514),
                ('霸王城址', 113.4687, 34.9477), ('虎牢关', 113.19, 34.84)]


def water_corners(g):
    """The 黄河, wider than the canal."""
    return g.polyline_distance([YELLOW_RIVER], offset=0.0) < YELLOW_HALF_WIDTH


def plateau(g):
    """(mountain mask, cells that must stay low, cells that must stay mountain): the hills above PLATEAU_ELEV
    (lower around 虎牢), smoothed, with the 虎牢 defile and every town, field and building site cut out; low pockets
    that only aircraft could reach join the mountain."""
    def grow(m, k=1):
        for _ in range(k):
            m = m | g.shift(m, 1, 0, False) | g.shift(m, -1, 0, False) | g.shift(m, 0, 1, False) | g.shift(m, 0, -1, False)
        return m

    passes = g.polyline_distance(PASSES.values()) < PASS_HALF_WIDTH
    keep = passes.copy()
    sites = [(lon, lat, 8) for _, lon, lat in STARTS]
    sites += [(lon, lat, 4) for _, lon, lat in ORE_TOWNS] + [(lon, lat, 3.5) for _, lon, lat in GEM_TOWNS]
    sites += [(lon, lat, max(w, h) + 2) for _, _, lon, lat, w, h in TECH + LANDMARKS]
    for lon, lat, r in sites:
        keep |= g.dist_to_point(lon, lat) < r
    keep |= grow(g.water_cell, 6)                    # room to step down from the cliff foot (2) to the water (0)
    ridge = np.zeros_like(keep)
    lon0, lat0, elev = PLATEAU_ELEV_HULAO
    thresh = np.where((g.CELL_LON < lon0) & (g.CELL_LAT > lat0), elev, PLATEAU_ELEV)
    hl, ha, a_km, b_km, _ = GUANGWU
    ridge_zone = np.hypot((g.CELL_LON - hl) * 111.32 * np.cos(np.radians(ha)) / a_km,
                          (g.CELL_LAT - ha) * 111.32 / b_km) < 1.3
    mount = (g.ELEV_CELL > thresh) & g.VALID & ~ridge_zone      # the raised 广武山 stays open ground
    mount = ~grow(~grow(mount, 2), 2)                 # close: no slots narrower than ~4 cells
    mount = grow(~grow(~mount, 2), 2)                 # open: no ridges narrower than ~4 cells
    mount &= ~keep
    from cliffs import majority
    mount = majority(majority(mount, 2), 3) & ~keep   # a smooth outline the cliff pieces can follow
    low = g.VALID & ~mount & ~g.water_cell
    lab, sizes = g.components(low)
    seeds = passes.copy()                             # land joined to a start, a pass or a ford / bridge
    for _, lon, lat in STARTS:
        seeds |= g.dist_to_point(lon, lat) < 3
    for _, lon, lat in BRIDGES:
        seeds |= g.dist_to_point(lon, lat) < 8
    reach = np.unique(lab[seeds & low])
    mount |= low & ~np.isin(lab, reach)
    lab, sizes = g.components(mount)
    mount &= (lab >= 0) & (sizes[np.maximum(lab, 0)] >= 60)
    return mount, keep, ridge


def ground_cover(g):
    """Autumn on the loess and the plain: stubble and millet fields, bare loess on the slopes, willows along the
    rivers and the canal, oak and pine on the 嵩山."""
    E, R, lon, lat, n1, n2 = g.ELEV_CELL, g.RELIEF_CELL, g.CELL_LON, g.CELL_LAT, g.NOISE, g.NOISE2
    cover = np.full(E.shape, g.GRASS)
    plain = E < 120
    loess = (E >= 120) & (E < 265)
    hills = E >= 265
    riparian = g.RIVER_D < 1.3
    canal = g.polyline_distance([HONGGOU]) < 3
    cover[plain & (n1 > 0.42)] = g.THICK                       # millet and stubble
    cover[plain & (n1 <= 0.42) & (n2 > 0.78)] = g.SAND         # threshing floors, sandy flats by the 黄河
    cover[loess & (n1 > 0.38)] = g.SAND                         # 黄土 gullies of the 邙山 and 广武山
    cover[loess & (n1 <= 0.38) & (n2 > 0.55)] = g.ROUGH
    cover[hills & (n2 > 0.5)] = g.ROUGH
    p = np.full(E.shape, 0.006)
    p[plain] = 0.008
    p[loess] = 0.015 + 0.02 * n2[loess]
    p[hills] = 0.08 + 0.25 * n1[hills]                          # oak and pine on the 嵩山 / 浮戏山
    p[riparian] = 0.10
    p[canal] = np.maximum(p[canal], 0.06)                       # willows along the 鸿沟
    road = g.polyline_distance(PASSES.values())
    p[road < PASS_HALF_WIDTH + 1] *= 0.3                       # keep the defile open for armies
    p[road < 2.5] = 0
    return cover, loess | hills, p


# --- buildings ---------------------------------------------------------------------------
from regions.pools import POOLS as CIVILIAN_POOLS   # noqa: E402

CIVILIAN_START_CLEAR = 7
PROP_DENSITY = 0.2
CIVILIAN_ZONES = [  # lon, lat, radius (cells), pool, density: villages and farms only
    (113.50, 34.852, 2, 'village', 0.5),        # 荥阳 (古荥)
    (114.555, 34.70, 2, 'village', 0.4),         # 陈留
    (113.95, 34.745, 2, 'village', 0.4),         # 中牟
    (113.70, 34.425, 2, 'village', 0.4),         # 新郑
    (113.80, 34.62, 5, 'farm', 0.25),            # 汉 farmland
    (114.05, 34.55, 5, 'farm', 0.25),
    (114.20, 35.00, 5, 'farm', 0.25),            # 楚 farmland
    (114.60, 34.85, 4, 'farm', 0.25),
    (113.25, 35.05, 5, 'farm', 0.25),            # 河内
]
CIVILIAN_FILL = None
