"""Chu-Han 5: Battle of the Wei River (潍水之战, November 203 BC). After taking 齐 (历下, 临淄) 韩信 met 龙且's
200,000 楚-齐 army across the 潍水 near 高密. He dammed the river upstream with ten thousand sandbags (囊沙壅水),
crossed the shallows, feigned a retreat and broke the dam while half of 龙且's army was in the river bed; 龙且 was
killed and the 齐王 田广 fled to 城阳 (莒). The 潍水 runs through the middle of the flat 胶莱 plain as real water
with a few fords; the 鲁中 hills (沂山, 穆陵关 on the 齐长城) close the south-west, 莱州湾 the north.
Towns are the 齐 / Western Han sites (the coordinates of the ancient cities are estimates near the modern towns);
the 峡山 reservoir and other modern reservoirs and canals (胶莱运河) are left out. River courses come from
OpenStreetMap (data/osm_chuhan5_weishui/water.json)."""
import numpy as np

from regions import osmwater

NAME = 'Chu-Han 5: Battle of the Wei River'
OUTFILE = 'chuhan5_weishui.mpr'
SEED = 203  # 203 BC
W, H = 164, 195
LX, LY, LW, LH = 2, 4, 160, 182
LON0, LON1, LAT0, LAT1 = 118.25, 120.15, 37.25, 35.4997
DEM_DIR, DEM_ZOOM = 'data/dem_chuhan5_weishui', 10
BLUR_SMOOTH, BLUR_WIDE, BLUR_TERRAIN = 1, 6, 2
ELEV_PTS = [0, 35, 60, 100, 160, 250, 400, 600, 950]   # the 胶莱 plain stays level 0-1; 沂山 rises to 8
LVL_PTS = [0, 0, 1, 1.6, 2.4, 3.4, 5, 6.5, 8]
LIGHTING = dict(Red=1.03, Green=0.99, Blue=0.92, Level=0.02, Ambient=1.0)   # late autumn, low sun

STARTS = [  # 汉 (韩信) on the west bank, 楚-齐 (龙且, 田广) on the east bank and in the south
    ('临淄', 118.350, 36.860),       # 齐都, taken by 韩信 before the battle
    ('平寿', 119.100, 36.660),       # 汉军 rear on the 潍水 road
    ('潍水西岸', 119.300, 36.520),    # 韩信's camp facing the river
    ('高密', 119.620, 36.340),       # 龙且's camp on the east bank
    ('即墨', 119.970, 36.630),       # 胶东
    ('夜邑', 119.980, 37.130),       # 掖, on 莱州湾
    ('东武', 119.440, 35.960),       # upper 潍水, east bank
    ('城阳', 118.830, 35.590),       # 莒: where 田广 fled
]
START_FLAT, START_ORE_DIST, START_WATER_CLEAR = 9, 7.5, 3
MAP_CENTER = (119.30, 36.45)
ORE_TOWNS = [('剧', 118.750, 36.780), ('寿光', 118.740, 36.900), ('昌乐', 118.830, 36.700), ('营陵', 118.980, 36.600),
             ('淳于', 119.270, 36.560), ('安丘', 119.200, 36.430), ('都昌', 119.380, 36.860), ('下密', 119.560, 36.820),
             ('夷安', 119.760, 36.390), ('黔陬', 119.800, 36.160), ('计斤', 120.000, 36.290), ('当利', 119.850, 37.030),
             ('卢乡', 120.050, 36.880), ('胶阳', 119.700, 36.580), ('姑幕', 119.250, 36.120), ('平昌', 119.050, 36.250),
             ('朱虚', 118.800, 36.470), ('临朐', 118.540, 36.510), ('东安', 118.630, 35.780), ('柜', 119.950, 35.880),
             ('石泉', 119.600, 36.050), ('东安平', 118.450, 36.820), ('广饶', 118.420, 37.050), ('高苑', 118.550, 37.000),
             ('胶东北', 119.800, 36.800), ('五莲', 119.200, 35.760), ('莒北', 118.950, 35.800), ('诸城东', 119.600, 35.900)]
GEM_TOWNS = [('双王城盐场', 118.760, 37.080),     # 齐's sea salt on 莱州湾 (鱼盐之利)
             ('穆陵关', 118.790, 36.060),         # the 齐长城 gate in the 沂山 hills
             ('琅琊', 119.880, 35.700),           # 琅琊台, where 秦始皇 set up his stone
             ('大泽山', 119.990, 36.960)]
OIL = []
TECH = [('CAOUTP', '韩信坝 (囊沙壅水)', 119.415, 36.395, 4, 3),   # the sandbag dam upstream of the crossing
        ('CAOUTP', '龙且营', 119.520, 36.470, 4, 3),
        ('CAHOSP', '稷下学宫', 118.380, 36.790, 6, 4)]
FORDS = [  # shallows of the 潍水, north to south: no bridges in 203 BC, the river is cut by a short stretch of dry ground
    ('下密渡', 119.437, 36.860),
    ('淳于渡', 119.392, 36.625),
    ('韩信渡', 119.426, 36.490),     # where half of 龙且's army was caught when the dam broke
    ('东武渡', 119.373, 36.050),
]
FORD_KM = 3.5                        # half the length of river left dry at a ford (the water is laid in 2x2 blocks)
CITIES = [('临淄', 118.350, 36.860, 3), ('即墨', 119.970, 36.630, 1.5), ('莒', 118.830, 35.590, 1.5),
          ('高密', 119.620, 36.340, 1.2), ('平寿', 119.100, 36.660, 1.2), ('东武', 119.420, 35.975, 1.2),
          ('夜邑', 119.980, 37.130, 1.2)]
CITY_COVER = 0   # walled towns of rammed earth, not paved city blocks
TRACKS = {  # roads of 齐 and the 秦 imperial tours (dirt)
    '齐东道': [(118.350, 36.860), (118.750, 36.780), (118.980, 36.600), (119.100, 36.660), (119.270, 36.560),
             (119.420, 36.490), (119.620, 36.340), (119.800, 36.160), (120.000, 36.290), (120.150, 36.300)],
    '北海道': [(118.350, 36.860), (118.740, 36.900), (119.380, 36.860), (119.560, 36.820), (119.850, 37.030),
             (119.980, 37.130), (120.150, 37.200)],
    '平寿-都昌': [(119.100, 36.660), (119.380, 36.860)],
    '即墨道': [(119.980, 37.130), (120.050, 36.880), (119.970, 36.630), (119.760, 36.390), (119.620, 36.340)],
    '淳于-胶东': [(119.270, 36.560), (119.400, 36.630), (119.700, 36.580), (119.970, 36.630)],
    '穆陵道': [(118.350, 36.860), (118.540, 36.510), (118.790, 36.060), (118.630, 35.780), (118.830, 35.590)],
    '琅琊道': [(119.200, 36.430), (119.250, 36.120), (119.360, 36.000), (119.600, 35.900), (119.880, 35.700)],
    '东武-高密': [(119.360, 36.000), (119.440, 36.050), (119.600, 36.050), (119.620, 36.340)],
    '东武-莒': [(119.360, 36.000), (119.200, 35.760), (118.950, 35.800), (118.830, 35.590)],
}
TRACK_COVER = 'SAND'
ROADS = dict(TRACKS)
RIVER_NAMES = set()
RIVER_BBOX = (118.0, 35.3, 120.4, 37.5)
WATER_SRC = 'data/osm_chuhan5_weishui/water.json'
WEI = '潍河'
TRIBUTARIES = {'汶河', '渠河', '弥河', '巨洋河', '淄河', '白浪河', '胶河', '虞河', '丹河', '浯河', '百尺河', '大沽河',
               '小沽河', '扶淇河', '史角河', '沂河', '沭河', '涨湎河', '温泉河'}   # not 小清河 (Song) or 胶莱河 (Yuan canal)
WATER_HALF_WIDTH = 0.9
LAKES = []
WATER_LEVEL = 0
MIN_WATER_CORNERS, MIN_LAND_CORNERS = 6, 8
MOUNTAIN_TREES = ['TREE05', 'TREE06', 'TREE07', 'TREE08', 'TREE15', 'TREE25', 'TREE26', 'TREE27']
VALLEY_TREES = ['TREE01', 'TREE02', 'TREE03', 'TREE04', 'TREE10', 'TREE11', 'TREE12', 'TREE17']


def _wei_lines():
    """The 潍水 below 东武 as water, broken at the fords; its headwaters in the hills stay a creek."""
    lines = []
    for line in osmwater.lines(WATER_SRC, {WEI}):
        pts = []
        for (a, b), (c, d) in zip(line, line[1:]):     # resample every ~100 m so the cuts land where they should
            n = max(1, int(np.hypot((c - a) * 89.7, (d - b) * 111.3) / 0.1))
            pts += [(a + (c - a) * k / n, b + (d - b) * k / n) for k in range(n)]
        lines.append(pts + [line[-1]])
    allp = np.array([p for pts in lines for p in pts])
    fords = []                                         # each ford snapped onto the river
    for _, fx, fy in FORDS:
        i = np.argmin(np.hypot((allp[:, 0] - fx) * 89.7, (allp[:, 1] - fy) * 111.3))
        fords.append(allp[i])
    out = []
    for pts in lines:
        part = []
        for x, y in pts:
            dry = x < 119.27 or y <= 35.97 or any(np.hypot((x - fx) * 89.7, (y - fy) * 111.3) < FORD_KM for fx, fy in fords)
            if dry:
                if len(part) > 1:
                    out.append(part)
                part = []
            else:
                part.append((x, y))
        if len(part) > 1:
            out.append(part)
    return out


WATER_RIVERS = _wei_lines()
EXTRA_RIVERS = osmwater.lines(WATER_SRC, TRIBUTARIES) + \
    [[(x, y) for x, y in line if x < 119.28 or y <= 35.98] for line in osmwater.lines(WATER_SRC, {WEI})]
EXTRA_RIVERS = [l for l in EXTRA_RIVERS if len(l) > 1]


def water_corners(g):
    """莱州湾 and the Yellow Sea from the bathymetry (open sea only)."""
    sea = g.sample(g.box_blur(g.DEM, 1), g.CORNER_LON, g.CORNER_LAT) < -0.5
    lab, sizes = g.components(sea)
    return (lab >= 0) & (sizes[lab] > 3000)


def ground_cover(g):
    E, R, lon, lat, n1, n2 = g.ELEV_CELL, g.RELIEF_CELL, g.CELL_LON, g.CELL_LAT, g.NOISE, g.NOISE2
    cover = np.full(E.shape, g.GRASS)
    p = np.full(E.shape, 0.01)
    hills = (E > 140) | (R > 30)                               # 沂山, 五莲山, 大泽山
    plain = ~hills
    riparian = (g.RIVER_D < 1.3) & plain
    coast = plain & (E < 8)                                    # 莱州湾 salt flats

    cover[plain & (n1 > 0.45)] = g.THICK                       # harvested millet and wheat stubble
    cover[hills & (n2 > 0.5)] = g.ROUGH                        # bare rock of the 鲁中 hills
    cover[coast & (n1 > 0.62)] = g.SAND                        # salt pans and tidal sand
    cover[coast & (n1 <= 0.62)] = g.THICK                      # salt marsh
    p[plain] = 0.008
    p[hills] = 0.06 + 0.14 * (1 - n1[hills])                   # oak and pine on the hills
    p[riparian] = 0.09                                         # willows along the rivers
    p[coast] = 0.0
    return cover, hills, p


# --- buildings ---------------------------------------------------------------------------
from regions.pools import POOLS as CIVILIAN_POOLS   # noqa: E402

CIVILIAN_START_CLEAR = 7
PROP_DENSITY = 0.2
CIVILIAN_ZONES = [  # lon, lat, radius (cells), pool, density: farming villages of 齐 and 胶东
    (118.75, 36.70, 5, 'farm', 0.25),
    (119.05, 36.45, 5, 'farm', 0.25),
    (119.25, 36.75, 5, 'village', 0.25),
    (119.75, 36.50, 5, 'farm', 0.25),
    (119.85, 36.90, 5, 'village', 0.25),
    (119.55, 36.15, 5, 'farm', 0.25),
    (118.95, 35.95, 4, 'village', 0.25),
    (118.55, 36.95, 4, 'farm', 0.25),
]
