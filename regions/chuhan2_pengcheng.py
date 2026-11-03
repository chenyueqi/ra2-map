"""楚汉争霸 2: Battle of Pengcheng (彭城之战, April 205 BC). 刘邦 took the 楚 capital 彭城 with 56万 allied troops;
项羽 rode back from 齐 with 3万 cavalry by way of 鲁 and 胡陵, struck from 萧 in the west at dawn, drove the Han army
into the 谷水 and 泗水, chased it south into the hills and broke it again on the 睢水 east of 灵壁 ("睢水为之不流").

The rivers are drawn on their courses of the time: the 泗水 from 胡陵 past 沛, 留 and 彭城, then by the 吕梁 rapids to
下邳 and 下相; the 获水 (the lower 汴水; some commentators take the 谷水 of the 史记 to be its last reach) from 下邑 and
萧 into the 泗水 at 彭城; the 睢水 from 芒 past 相, 竹邑, 取虑 and 睢陵 to 下相. All three are open water with a few
crossings. The 废黄河 (the Yellow River's bed of 1194-1855, which later took over the lower 泗水) and its levees did
not exist; the whole plain maps to one level so the levee ridge in today's DEM disappears, and 微山湖 / 骆马湖 (lakes
of the Ming and Qing) are dry farmland. Positions of ancient towns are put on their modern successors (OSM,
data/osm_chuhan2_pengcheng/places.json, read only while writing this file) or estimated."""
import numpy as np

NAME = 'Chu-Han 2: Battle of Pengcheng'
OUTFILE = 'chuhan2_pengcheng.mpr'
SEED = 205  # April 205 BC
W, H = 200, 157
LX, LY, LW, LH = 2, 4, 196, 144
LON0, LON1, LAT0, LAT1 = 116.05, 118.45, 35.02, 33.5534   # square cells: 0.563 km per iso unit both ways
DEM_DIR, DEM_ZOOM = 'data/dem_chuhan2_pengcheng', 10
BLUR_SMOOTH, BLUR_WIDE, BLUR_TERRAIN = 1, 6, 1
ELEV_PTS = [0, 62, 85, 120, 170, 240, 350]   # the plain (20-60 m) is one level; the hills of 彭城, 萧 and 吕梁 rise
LVL_PTS = [0, 0, 1, 2, 3, 4, 5]
LIGHTING = dict(Red=1.0, Green=1.01, Blue=0.97, Level=0.03, Ambient=1.02)   # spring on the 黄淮 plain
CITY_COVER = 493          # rammed-earth towns: dirt, not pavement
PROP_DENSITY = 0.0        # no street lights in 205 BC

STARTS = [
    ('彭城', 117.14, 34.27),        # 西楚 capital, taken by 刘邦; 项羽's target
    ('萧', 116.94, 34.165),         # 项羽 struck from 萧 at dawn, from the west
    ('下邑', 116.40, 34.44),        # 吕泽 held 下邑; 刘邦 gathered the routed army here
    ('沛', 116.93, 34.76),          # 刘邦's home county (丰沛)
    ('胡陵', 116.71, 34.90),        # 项羽's approach from 鲁 by 胡陵
    ('灵壁', 117.30, 33.93),        # "至灵壁东睢水上" (estimated site)
    ('下相', 118.20, 33.85),        # 项羽's birthplace, where the 睢水 meets the 泗水
    ('符离', 116.972, 33.754),      # south of the 睢水
]
START_FLAT, START_ORE_DIST, START_WATER_CLEAR = 8, 7.5, 3
MAP_CENTER = (117.2, 34.25)
ORE_TOWNS = [('丰', 116.653, 34.697), ('留', 117.02, 34.60), ('广戚', 117.10, 34.82), ('薛', 117.25, 34.96),
             ('傅阳', 117.68, 34.55), ('武原', 117.80, 34.42), ('下邳', 117.90, 34.20), ('吕', 117.34, 34.25),
             ('司吾', 118.30, 34.32), ('良成', 118.12, 34.46), ('取虑', 117.72, 33.79), ('睢陵', 117.94, 33.99),
             ('凌', 118.38, 34.08), ('僮', 118.15, 33.66), ('蕲', 117.09, 33.60), ('竹邑', 117.02, 33.80),
             ('相', 116.79, 34.00), ('芒', 116.45, 33.97), ('砀', 116.36, 34.20), ('杼秋', 116.72, 34.42),
             ('酂', 116.20, 33.95), ('单父', 116.14, 34.79), ('建成', 116.50, 34.60), ('甾丘', 117.45, 33.98),
             ('夏丘', 117.88, 33.64), ('傅阳南', 117.55, 34.40), ('彭城南', 117.35, 34.05)]
GEM_TOWNS = [('芒砀山', 116.39, 34.165),       # 刘邦 hid in the 芒砀 marshes and hills
             ('皇藏峪', 117.02, 34.06),        # the hills south of 萧 where the routed Han army fled ("汉卒皆南走山")
             ('大泽乡', 117.091, 33.59),       # 陈胜 and 吴广, 209 BC
             ('圯桥', 117.93, 34.19),          # 下邳: 张良 and the old man on the bridge
             ('吕梁洪', 117.42, 34.20)]        # the 泗水 rapids below 彭城
OIL = []
TECH = [('CAOUTP', '九里山', 117.11, 34.385, 4, 3),      # the hill north of 彭城
        ('CAOUTP', '戏马台', 117.22, 34.215, 4, 3),      # 项羽's review terrace south of 彭城
        ('CAHOSP', '下邳', 117.86, 34.27, 6, 4)]
CITIES = [('彭城', 117.181, 34.267, 2.2), ('下邳', 117.88, 34.22, 1.2), ('萧', 116.93, 34.20, 1.0),
          ('沛', 116.931, 34.762, 1.1), ('下相', 118.24, 33.90, 1.0), ('相', 116.79, 33.99, 1.0),
          ('丰', 116.653, 34.697, 0.9), ('符离', 116.972, 33.754, 0.9)]
WATER_RIVERS = [
    # 泗水: 胡陵 - 沛 - 留 - 彭城 - 吕梁 - 下邳 - 下相 (now under 微山湖 and the 废黄河)
    [(116.76, 35.06), (116.84, 34.95), (116.92, 34.86), (117.00, 34.76), (117.05, 34.66), (117.07, 34.58),
     (117.12, 34.48), (117.17, 34.38), (117.20, 34.31), (117.24, 34.25), (117.33, 34.19), (117.45, 34.155),
     (117.60, 34.125), (117.75, 34.12), (117.88, 34.13), (118.02, 34.07), (118.15, 34.00), (118.27, 33.96),
     (118.38, 33.88), (118.50, 33.79)],
    # 获水 (lower 汴水): 下邑 - 萧 - 彭城, into the 泗水
    [(116.00, 34.34), (116.30, 34.345), (116.55, 34.31), (116.75, 34.265), (116.93, 34.245), (117.05, 34.295),
     (117.13, 34.325), (117.185, 34.335)],
    # 睢水: 芒 - 相 - 竹邑 - 灵壁东 - 取虑 - 睢陵 - 下相
    [(116.00, 34.13), (116.30, 34.09), (116.55, 34.03), (116.80, 33.945), (116.95, 33.88), (117.10, 33.845),
     (117.30, 33.83), (117.50, 33.85), (117.70, 33.865), (117.93, 33.915), (118.10, 33.94), (118.265, 33.96)],
]
WATER_HALF_WIDTH = 0.9
LAKES = []
WATER_LEVEL = 0
MIN_WATER_CORNERS, MIN_LAND_CORNERS = 6, 8
BRIDGES = [  # bridges and fords of the time
    ('泗水 沛', 117.00, 34.76), ('泗水 留', 117.09, 34.53), ('泗水 彭城', 117.21, 34.29), ('泗水 吕梁', 117.38, 34.175),
    ('泗水 下邳', 117.86, 34.13), ('泗水 下相', 118.21, 33.98),
    ('获水 下邑', 116.42, 34.335), ('获水 萧', 116.85, 34.255), ('获水 彭城西', 117.08, 34.31),
    ('睢水 相', 116.75, 33.96), ('睢水 竹邑', 117.03, 33.86), ('睢水 灵壁东', 117.40, 33.84),
    ('睢水 取虑', 117.78, 33.88), ('睢水 睢陵', 118.02, 33.93),
]
TRACKS = {  # roads of the 秦 (dirt), crossing the rivers at the bridges
    '彭城-萧-砀': [(117.14, 34.27), (117.08, 34.31), (116.93, 34.21), (116.70, 34.22), (116.42, 34.335),
                 (116.40, 34.44), (116.05, 34.47)],
    '彭城-留-沛-胡陵': [(117.17, 34.30), (117.21, 34.29), (117.15, 34.45), (117.09, 34.53), (117.03, 34.68),
                   (117.00, 34.76), (116.93, 34.76), (116.71, 34.90), (116.60, 35.02)],
    '彭城-吕-下邳-下相': [(117.21, 34.29), (117.34, 34.25), (117.55, 34.22), (117.86, 34.20), (117.86, 34.13),
                    (118.00, 34.02), (118.21, 33.98), (118.20, 33.85)],
    '彭城-灵壁-符离': [(117.14, 34.27), (117.18, 34.15), (117.25, 34.00), (117.30, 33.93), (117.40, 33.84),
                  (117.25, 33.76), (116.972, 33.754), (116.97, 33.56)],
    '萧-相-竹邑': [(116.94, 34.165), (116.80, 34.05), (116.75, 33.96), (116.85, 33.90), (117.03, 33.86),
               (117.02, 33.80), (116.972, 33.754)],
    '沛-丰-单父': [(116.93, 34.76), (116.653, 34.697), (116.40, 34.70), (116.14, 34.79)],
}
TRACK_COVER = 'SAND'
ROADS = dict(TRACKS)
RIVER_NAMES = set()
RIVER_BBOX = (115.9, 33.4, 118.6, 35.1)
EXTRA_RIVERS = [  # smaller streams as tree belts
    [(116.20, 34.62), (116.45, 34.55), (116.70, 34.47), (116.95, 34.40), (117.12, 34.36)],    # 丰 - 杼秋 stream
    [(117.62, 34.62), (117.70, 34.45), (117.75, 34.30), (117.80, 34.15)],                     # 武原水 to the 泗水
    [(117.20, 33.60), (117.35, 33.70), (117.50, 33.80)],                                       # 蕲 stream to the 睢水
]
MOUNTAIN_TREES = ['TREE05', 'TREE06', 'TREE07', 'TREE08', 'TREE15', 'TREE01', 'TREE02', 'TREE03']
VALLEY_TREES = ['TREE01', 'TREE02', 'TREE03', 'TREE04', 'TREE10', 'TREE11', 'TREE12', 'TREE17']


def ground_cover(g):
    E, R, lon, lat, n1, n2 = g.ELEV_CELL, g.RELIEF_CELL, g.CELL_LON, g.CELL_LAT, g.NOISE, g.NOISE2
    cover = np.full(E.shape, g.GRASS)
    hills = (E > 70) | (R > 18)
    plain = ~hills
    riparian = (g.RIVER_D < 1.5) & plain
    marsh = plain & (np.hypot((lon - 116.42) / 0.12, (lat - 34.10) / 0.08) < 1)          # the 芒砀 marshes
    cover[plain & (n1 > 0.45)] = g.THICK                     # spring wheat and millet
    cover[plain & (n1 <= 0.45) & (n2 > 0.7)] = g.SAND        # bare loess fields and threshing floors
    cover[hills & (n2 > 0.5)] = g.ROUGH                      # rocky limestone hills
    cover[marsh & (n1 > 0.35)] = g.THICK
    p = np.full(E.shape, 0.012)
    p[hills] = 0.16 + 0.14 * (1 - n1[hills])                 # wooded hills (the 汉卒 fled into them)
    p[riparian] = 0.07
    p[marsh] = 0.09
    return cover, hills, p


# --- buildings ---------------------------------------------------------------------------
from regions.pools import POOLS as CIVILIAN_POOLS   # noqa: E402

CIVILIAN_START_CLEAR = 7
FLAT_CITIES = []
CIVILIAN_ZONES = [  # lon, lat, radius (cells), pool, density: farming villages only
    (116.65, 34.70, 3, 'village', 0.35),         # 丰
    (117.88, 34.24, 2.5, 'village', 0.4),        # 下邳
    (116.80, 34.00, 2.5, 'village', 0.35),       # 相
    (116.70, 34.55, 5, 'farm', 0.25),
    (117.50, 34.45, 5, 'farm', 0.25),
    (117.60, 34.00, 5, 'farm', 0.25),
    (116.60, 33.85, 5, 'farm', 0.25),
    (118.05, 34.30, 4, 'farm', 0.25),
    (117.50, 33.70, 4, 'farm', 0.25),
]
CIVILIAN_FILL = ('small', 0.4)
