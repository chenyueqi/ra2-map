"""Taiyuan Campaign (太原战役, Oct 1948 - 24 Apr 1949): 表里山河. The Taiyuan - Jinzhong basin of the Fen River is
walled by the Lüliang and Taihang ranges, made a plateau of cliffs that ground units cannot climb. Four passes
lead in: north over 石岭关 (同蒲北段) from the Xinzhou basin, east along the 正太 railway from 寿阳, south down the
Fen gorge below 介休 (the road from 临汾 that 徐向前's army took in the 晋中战役) and west up the Fen gorge to the
古交 coalfield. East of the city the four forts of 东山 (小窑头, 牛驼寨, 山头, 淖马), where the decisive fighting of
Oct - Nov 1948 took place, sit on low foothills in front of the cliffs."""
import numpy as np

NAME = 'Taiyuan Campaign'
OUTFILE = 'taiyuan.mpr'
SEED = 1949  # Taiyuan fell on 24 Apr 1949
W, H = 160, 195
LX, LY, LW, LH = 2, 4, 156, 182
LON0, LON1, LAT0, LAT1 = 111.65, 113.40, 38.545, 36.9215
DEM_DIR, DEM_ZOOM = 'data/dem_taiyuan', 9
BLUR_SMOOTH, BLUR_WIDE, BLUR_TERRAIN = 1, 5, 2
ELEV_PTS = [700, 820, 950, 1080]                  # basin floor 750-800 m at level 0, 寿阳 (1050 m) at level 2
LVL_PTS = [0, 0, 1, 2]
CLIFF_FOOT = 2                                    # every cliff stands on level 2; its top is level 6
PLATEAU_ELEV = 1000                               # the mountain mask follows this contour (m) ...
PLATEAU_ELEV_SHOUYANG = (113.06, 37.81, 1150)    # ... and this one NE of (113.06 E, 37.81 N): the 寿阳 basin lies at 1050 m
PLATEAU_ELEV_PTS = [1000, 1400, 1750, 2050, 2450]
PLATEAU_LVL_PTS = [6, 8, 10, 12, 13]              # 关帝山 / 云中山 up to level 13
PLATEAU_TREES = 0.0
LIGHTING = dict(Red=1.02, Green=1.0, Blue=0.96, Level=0.03, Ambient=1.03)   # dusty spring light on the loess

PASSES = {  # the only ground routes into the basin (lon, lat along the railways and the Fen)
    '北 石岭关': [(112.714, 38.078), (112.723, 38.101), (112.705, 38.156), (112.62, 38.235), (112.651, 38.283),
                 (112.714, 38.362), (112.75, 38.397)],
    '东 正太线': [(112.733, 37.689), (112.843, 37.686), (112.988, 37.737), (113.049, 37.736), (113.073, 37.80),
                 (113.153, 37.84), (113.176, 37.881)],
    '南 汾河峡口': [(111.93, 37.03), (111.881, 37.007), (111.823, 36.988), (111.801, 36.947), (111.79, 36.90)],
    '西 汾河峡谷': [(112.45, 37.985), (112.40, 38.005), (112.36, 37.97), (112.31, 37.94), (112.27, 37.925),
                  (112.18, 37.91), (112.13, 37.915), (112.07, 37.94), (112.02, 37.985), (111.97, 38.03),
                  (111.90, 38.06), (111.81, 38.07)],
}
PASS_HALF_WIDTH = 3.5
RIDGES = {}                                       # always mountain, so that two passes never meet inside the range
RIDGE_HALF_WIDTH = 3.0

STARTS = [  # the eight fronts of 1948-49
    ('太原', 112.535, 37.862),         # 阎锡山's walled city
    ('阳曲', 112.67, 38.06),           # 北面: 黄樵松 / 城北
    ('忻州', 112.72, 38.42),           # beyond 石岭关
    ('寿阳', 113.17, 37.89),           # beyond the 正太 passes
    ('榆次', 112.70, 37.70),           # 正太 / 同蒲 junction, taken in the 晋中战役
    ('太谷', 112.55, 37.43),
    ('平遥', 112.18, 37.205),
    ('文水', 112.03, 37.432),          # 刘胡兰's home county
]
START_FLAT, START_ORE_DIST, START_WATER_CLEAR = 9, 7.5, 3
MAP_CENTER = (112.45, 37.70)
ORE_TOWNS = [('清徐', 112.354, 37.605), ('交城', 112.154, 37.555), ('祁县', 112.319, 37.357), ('孝义', 111.772, 37.145),
             ('介休', 111.915, 37.031), ('晋源', 112.471, 37.714), ('徐沟', 112.46, 37.53), ('汾阳', 111.80, 37.30),
             ('东阳', 112.67, 37.57), ('张兰', 112.04, 37.13), ('定襄', 112.954, 38.472), ('忻州南', 112.66, 38.33),
             ('寿阳东', 113.29, 37.86), ('寿阳北', 113.15, 38.00), ('娄烦', 111.84, 38.08), ('介休南', 111.83, 36.975),
             ('阳曲北', 112.72, 38.14), ('榆次东', 112.86, 37.69), ('平遥东', 112.28, 37.27), ('文水北', 112.07, 37.50)]
GEM_TOWNS = [('西山煤矿 (白家庄)', 112.45, 37.86),    # the coal pits west of the city
             ('小窑头', 112.612, 37.918),             # 东山四大要塞
             ('山头', 112.628, 37.866),
             ('古交煤田', 112.18, 37.905)]            # up the western Fen gorge
OIL = []
TECH = [('CAOUTP', '牛驼寨要塞', 112.608, 37.896, 4, 3),
        ('CAOUTP', '淖马要塞', 112.618, 37.838, 4, 3),
        ('CAAIRP', '武宿机场', 112.628, 37.755, 3, 3),    # its loss in late 1948 ended the airlift
        ('CAHOSP', '汾阳医院', 111.79, 37.245, 6, 4)]     # the American mission hospital
LANDMARKS = []
CITIES = [('太原', 112.555, 37.865, 3), ('榆次', 112.70, 37.70, 1.6), ('忻州', 112.73, 38.415, 1.5),
          ('平遥', 112.18, 37.205, 1.4), ('汾阳', 111.783, 37.268, 1.4), ('介休', 111.915, 37.031, 1.3),
          ('太谷', 112.545, 37.427, 1.3), ('祁县', 112.319, 37.357, 1.1), ('寿阳', 113.17, 37.89, 1.1),
          ('阳曲', 112.67, 38.058, 1), ('清徐', 112.354, 37.605, 1), ('交城', 112.154, 37.555, 1),
          ('文水', 112.031, 37.432, 1), ('孝义', 111.772, 37.145, 1), ('古交', 112.169, 37.907, 0.8)]
RAILWAYS = {  # Yan Xishan's metre-gauge 同蒲 and the 正太 line
    '同蒲北段': [(112.56, 37.855), (112.567, 37.97), (112.714, 38.078), (112.723, 38.101), (112.705, 38.156),
             (112.583, 38.249), (112.603, 38.302), (112.651, 38.283), (112.687, 38.295), (112.714, 38.362),
             (112.75, 38.397), (112.727, 38.538), (112.734, 38.56)],
    '正太线': [(112.56, 37.855), (112.645, 37.764), (112.733, 37.689), (112.843, 37.686), (112.988, 37.737),
            (113.049, 37.736), (113.073, 37.80), (113.153, 37.84), (113.176, 37.881), (113.276, 37.862),
            (113.42, 37.85)],
    '同蒲南段': [(112.733, 37.689), (112.743, 37.635), (112.72, 37.617), (112.681, 37.549), (112.675, 37.512),
             (112.593, 37.451), (112.557, 37.417), (112.509, 37.398), (112.411, 37.367), (112.316, 37.341),
             (112.269, 37.288), (112.204, 37.247), (112.169, 37.212), (112.093, 37.16), (111.922, 37.021),
             (111.852, 37.012), (111.823, 36.988), (111.801, 36.947), (111.78, 36.90)],
}
ROADS = dict(PASSES, **RAILWAYS, **{  # tree-free corridors along the old roads
    '汾阳-太原': [(111.783, 37.268), (112.031, 37.432), (112.154, 37.555), (112.354, 37.605), (112.471, 37.714),
              (112.555, 37.865)],
})
RIVER_NAMES = set()
RIVER_BBOX = (111.0, 36.5, 114.0, 39.0)
from regions import osmwater   # noqa: E402
EXTRA_RIVERS = osmwater.lines('data/osm_taiyuan/rivers.json', {'汾河', '潇河', '文峪河', '乌马河', '昌源河', '滹沱河',
                                                               '牧马河', '涂河', '松溪河'})
WATER_RIVERS, WATER_HALF_WIDTH, LAKES = [], 0, []
WATER_LEVEL = 0
MOUNTAIN_TREES = ['TREE25', 'TREE26', 'TREE27', 'TREE05', 'TREE06', 'TREE07', 'TREE08', 'TREE15']
VALLEY_TREES = ['TREE01', 'TREE02', 'TREE03', 'TREE04', 'TREE10', 'TREE11', 'TREE12', 'TREE17']
CHECK_POINTS = [('小窑头', 112.612, 37.918), ('牛驼寨', 112.608, 37.896), ('山头', 112.628, 37.866),
                ('淖马', 112.618, 37.838), ('古交', 112.18, 37.905), ('娄烦', 111.84, 38.08), ('介休南', 111.83, 36.975)]


def plateau(g):
    """(mountain mask, cells that must stay low, cells that must stay mountain): the land above PLATEAU_ELEV,
    smoothed, plus the RIDGES, with the passes and every town, field and building site cut out; low pockets
    that only aircraft could reach join the mountain."""
    def grow(m, k=1):
        for _ in range(k):
            m = m | g.shift(m, 1, 0, False) | g.shift(m, -1, 0, False) | g.shift(m, 0, 1, False) | g.shift(m, 0, -1, False)
        return m

    passes = g.polyline_distance(PASSES.values()) < PASS_HALF_WIDTH
    keep = passes.copy()
    sites = [(lon, lat, 8) for _, lon, lat in STARTS]
    sites += [(lon, lat, 4) for _, lon, lat in ORE_TOWNS] + [(lon, lat, 3.5) for _, lon, lat in GEM_TOWNS]
    sites += [(lon, lat, 3.5) for _, lon, lat in OIL]
    sites += [(lon, lat, max(w, h) + 2) for _, _, lon, lat, w, h in TECH + LANDMARKS]
    sites += [(lon, lat, r + 1) for _, lon, lat, r in FLAT_CITIES]
    for lon, lat, r in sites:
        keep |= g.dist_to_point(lon, lat) < r
    ridge = (g.polyline_distance(RIDGES.values()) < RIDGE_HALF_WIDTH) & ~keep if RIDGES else np.zeros_like(keep)
    lon0, lat0, elev = PLATEAU_ELEV_SHOUYANG
    mount = (g.ELEV_CELL > np.where((g.CELL_LON > lon0) & (g.CELL_LAT > lat0), elev, PLATEAU_ELEV)) & g.VALID
    mount = ~grow(~grow(mount, 2), 2)                 # close: no slots narrower than ~4 cells
    mount = grow(~grow(~mount, 2), 2)                 # open: no ridges narrower than ~4 cells
    mount = (mount | ridge) & ~keep
    from cliffs import majority
    mount = (majority(majority(mount, 2), 3) | ridge) & ~keep   # a smooth outline the cliff pieces can follow
    low = g.VALID & ~mount
    lab, sizes = g.components(low)
    border = g.VALID.copy()
    for rx, ry in g.CELLS:
        border[rx, ry] = not g.in_local(rx, ry, 3)
    reach = np.unique(lab[(passes | border) & low])
    mount |= low & ~np.isin(lab, reach)
    lab, sizes = g.components(mount)
    mount &= (lab >= 0) & (sizes[np.maximum(lab, 0)] >= 60)
    return mount, keep, ridge


def ground_cover(g):
    """Spring 1949 on the loess: green wheat in the basin, bare yellow loess on the hills, pine on the high ranges."""
    E, R, lon, lat, n1, n2 = g.ELEV_CELL, g.RELIEF_CELL, g.CELL_LON, g.CELL_LAT, g.NOISE, g.NOISE2
    cover = np.full(E.shape, g.GRASS)
    basin = E < 900
    loess = (E >= 900) & (E < 1700)
    high = E >= 1700
    riparian = g.RIVER_D < 1.3
    cover[basin & (n1 > 0.45)] = g.THICK                       # winter wheat coming up
    cover[basin & (n1 <= 0.45) & (n2 > 0.75)] = g.SAND         # dusty threshing floors and fallow
    cover[loess & (n1 > 0.35)] = g.SAND                         # 黄土 gullies
    cover[loess & (n1 <= 0.35) & (n2 > 0.5)] = g.ROUGH
    cover[high & (n2 > 0.55)] = g.ROUGH
    cover[riparian & (E < 1000)] = g.THICK
    p = np.full(E.shape, 0.006)
    p[basin] = 0.01
    p[riparian] = 0.12                                          # poplars and willows along the Fen
    p[loess] = 0.02 + 0.03 * n2[loess]
    p[high] = 0.15 + 0.35 * n1[high]                            # 关帝山 pine and larch
    road = g.polyline_distance(PASSES.values())
    p[road < PASS_HALF_WIDTH + 1] *= 0.3                       # keep the passes open for armies
    p[road < 2.5] = 0
    mountain = loess | high
    return cover, mountain, p


# --- buildings ---------------------------------------------------------------------------
from regions.pools import POOLS as CIVILIAN_POOLS   # noqa: E402

CIVILIAN_START_CLEAR = 7
FLAT_CITIES = [('太原', 112.555, 37.865, 3)]
PROP_DENSITY = 0.3
CIVILIAN_ZONES = [  # lon, lat, radius (cells), pool, density
    (112.585, 37.89, 2, 'industrial', 0.7),      # 太原兵工厂 / 西北实业公司
    (112.555, 37.86, 2.5, 'small', 0.8),         # the walled city
    (112.70, 37.70, 2, 'industrial', 0.5),       # 晋华纱厂, 榆次
    (112.18, 37.205, 1.6, 'village', 0.7),       # 平遥古城
    (112.545, 37.427, 1.5, 'village', 0.6),      # 太谷
    (112.319, 37.357, 1.5, 'village', 0.6),      # 祁县 merchant compounds
    (111.783, 37.268, 1.5, 'small', 0.6),        # 汾阳
    (111.915, 37.031, 1.5, 'small', 0.6),        # 介休
    (112.73, 38.415, 1.5, 'small', 0.6),         # 忻州
    (112.25, 37.45, 5, 'farm', 0.25),            # the Jinzhong plain
    (112.45, 37.60, 4, 'farm', 0.25),
    (111.95, 37.30, 4, 'farm', 0.25),
    (112.40, 37.30, 4, 'farm', 0.2),
    (112.85, 38.45, 4, 'farm', 0.25),            # Xinzhou basin
    (113.25, 37.90, 3, 'village', 0.3),          # 寿阳
]
CIVILIAN_FILL = ('small', 0.6)
