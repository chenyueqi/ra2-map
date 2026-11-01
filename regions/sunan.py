"""Southern Jiangsu + Shanghai (苏南+上海): Nanjing to the East China Sea, the Yangtze with its real bridges, Taihu."""
import numpy as np

from regions import osmwater

NAME = 'Sunan Shanghai'
OUTFILE = 'sunan.mpr'
SEED = 1843  # Shanghai opened as a treaty port
W, H = 188, 114
LX, LY, LW, LH = 2, 4, 184, 101
LON0, LON1, LAT0, LAT1 = 118.45, 122.00, 32.40, 30.72
DEM_DIR, DEM_ZOOM = 'data/dem_sunan', 9
BLUR_SMOOTH, BLUR_WIDE, BLUR_TERRAIN = 1, 5, 2
ELEV_PTS = [0, 15, 40, 80, 130, 190, 260, 350, 470, 650]
LVL_PTS = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8.5]
LIGHTING = dict(Red=0.95, Green=1.02, Blue=0.99, Level=0.03, Ambient=1.03)   # humid subtropical green

STARTS = [
    ('南京', 118.85, 31.90),
    ('镇江', 119.45, 32.10),
    ('常州', 119.90, 31.82),
    ('无锡', 120.36, 31.62),
    ('苏州', 120.58, 31.24),
    ('上海', 121.25, 31.08),
    ('南通', 120.90, 32.08),
    ('溧阳', 119.45, 31.42),
]
START_FLAT, START_ORE_DIST, START_WATER_CLEAR = 8, 7.0, 3
MAP_CENTER = (120.15, 31.65)
ORE_TOWNS = [('句容', 119.17, 31.95), ('丹阳', 119.57, 32.00), ('金坛', 119.58, 31.74), ('江阴', 120.28, 31.84),
             ('张家港', 120.55, 31.86), ('常熟', 120.75, 31.65), ('昆山', 120.98, 31.38), ('吴江', 120.65, 31.05),
             ('宜兴', 119.82, 31.36), ('溧水', 119.03, 31.65), ('高淳', 118.88, 31.33), ('扬州', 119.42, 32.33),
             ('泰兴', 120.02, 32.17), ('仪征', 119.18, 32.27), ('嘉定', 121.25, 31.38), ('湖州', 120.09, 30.88),
             ('太仓', 121.10, 31.50), ('启东', 121.65, 31.80), ('如皋', 120.57, 32.33), ('靖江', 120.27, 32.02),
             ('崇明', 121.40, 31.65), ('嘉兴', 120.78, 30.80), ('浦东', 121.70, 31.05), ('六合', 118.83, 32.33)]
GEM_TOWNS = [('茅山', 119.33, 31.77), ('宜兴南山', 119.72, 31.20), ('马山', 120.10, 31.43), ('佘山', 121.19, 31.10),
             ('东山', 120.40, 31.08), ('汤山', 119.05, 32.05)]
OIL = [('金陵石化', 118.92, 32.15), ('扬子石化', 118.78, 32.25), ('上海石化', 121.33, 30.77),
       ('高桥石化', 121.58, 31.33), ('太仓港', 121.20, 31.62), ('镇江港', 119.70, 32.20)]
TECH = [('CAAIRP', '浦东机场', 121.80, 31.15, 3, 3),
        ('CAAIRP', '禄口机场', 118.87, 31.73, 3, 3),
        ('CAAIRP', '硕放机场', 120.43, 31.49, 3, 3),
        ('CAHOSP', '瑞金医院', 121.47, 31.21, 6, 4),
        ('CAOUTP', '吴淞口', 121.48, 31.38, 4, 3)]
LANDMARKS = [('CACHIG05', '上海中心大厦', 121.505, 31.235, 4, 4),
             ('CANEWY05', '紫峰大厦', 118.78, 32.06, 2, 2)]
BRIDGES = [  # Yangtze crossings, west to east (positions from OpenStreetMap)
    ('南京长江大桥', 118.738, 32.114),
    ('南京栖霞山长江大桥', 118.94, 32.178),
    ('润扬长江大桥', 119.367, 32.217),
    ('五峰山长江大桥', 119.63, 32.21),
    ('泰州长江大桥', 119.878, 32.248),
    ('江阴长江大桥', 120.27, 31.94),
    ('沪苏通长江大桥', 120.72, 32.013),
    ('苏通长江大桥', 120.995, 31.789),
    ('崇启大桥', 121.60, 31.78),
    ('上海长江大桥', 121.73, 31.47),
]
CITIES = [('上海', 121.47, 31.23, 7), ('浦东', 121.58, 31.22, 4), ('南京', 118.78, 32.05, 4.5), ('苏州', 120.62, 31.31, 4),
          ('无锡', 120.30, 31.57, 3.5), ('常州', 119.97, 31.78, 3.5), ('南通', 120.89, 32.01, 3), ('镇江', 119.45, 32.19, 2.5),
          ('扬州', 119.42, 32.39, 2.5), ('昆山', 120.98, 31.38, 2.5), ('江阴', 120.28, 31.91, 2), ('常熟', 120.75, 31.65, 2),
          ('张家港', 120.55, 31.87, 1.6), ('宜兴', 119.82, 31.36, 1.6), ('溧阳', 119.48, 31.42, 1.6), ('湖州', 120.09, 30.89, 2),
          ('嘉兴', 120.76, 30.76, 2), ('丹阳', 119.57, 32.00, 1.4), ('金坛', 119.58, 31.74, 1.2), ('句容', 119.17, 31.95, 1.2),
          ('松江', 121.23, 31.03, 2), ('嘉定', 121.25, 31.38, 1.8), ('吴江', 120.65, 31.16, 1.6), ('太仓', 121.10, 31.45, 1.4)]
ROADS = {
    'G42 沪宁': [(118.85, 32.03), (119.45, 32.15), (119.57, 32.00), (119.97, 31.78), (120.30, 31.57), (120.62, 31.32),
               (120.98, 31.38), (121.40, 31.25)],
    'G25 宁杭': [(118.85, 32.00), (119.03, 31.65), (119.48, 31.42), (119.82, 31.36), (120.09, 30.87)],
    '沿江': [(119.97, 31.78), (120.28, 31.85), (120.55, 31.87), (120.75, 31.65), (121.10, 31.45), (121.40, 31.30)],
    '苏嘉杭': [(120.62, 31.30), (120.65, 31.10), (120.75, 30.75)],
    'G2 京沪 north': [(120.27, 31.94), (120.27, 32.10), (120.30, 32.40)],
    '沪陕': [(121.00, 31.78), (120.89, 32.01), (120.57, 32.33)],
    '润扬': [(119.37, 32.215), (119.42, 32.39)],
    'G50 沪渝': [(121.40, 31.20), (121.23, 31.03), (120.65, 31.05), (120.09, 30.88)],
}
RIVER_NAMES = set()
RIVER_BBOX = (118.0, 30.5, 122.5, 32.7)
EXTRA_RIVERS = []


def _yangtze_upstream():
    """The Nanjing-Jiangyin reach is too narrow for the bathymetry: draw it from the OSM centreline."""
    return [[(x, y) for x, y in line if x <= 120.9] for line in osmwater.lines('data/osm_yangtze/yangtze.json', {'长江'})
            if any(x <= 120.9 and y > 31.5 for x, y in line)]


WATER_RIVERS = _yangtze_upstream() + [  # 长江北支: silted narrow, but it is what makes Chongming an island
    [(121.22, 31.80), (121.32, 31.835), (121.45, 31.81), (121.60, 31.785), (121.75, 31.765), (122.0, 31.73)]]
WATER_HALF_WIDTH = 1.3
LAKES = []
WATER_LEVEL = 0
MIN_WATER_CORNERS, MIN_LAND_CORNERS = 6, 8
MOUNTAIN_TREES = ['TREE01', 'TREE02', 'TREE03', 'TREE04', 'TREE10', 'TREE11', 'TREE12', 'TREE17']   # evergreen broadleaf and bamboo hills: the valley set, not the Rockies set
VALLEY_TREES = ['TREE01', 'TREE02', 'TREE03', 'TREE04', 'TREE10', 'TREE11', 'TREE12', 'TREE17']


def water_corners(g):
    """The estuary and the sea from the bathymetry; Taihu is too shallow for it, so its shore comes from Natural Earth."""
    sea = g.sample(g.box_blur(g.DEM, 1), g.CORNER_LON, g.CORNER_LAT) < -0.5
    return sea | osmwater.mask(g, osmwater.ne_lakes({'Tai Hu', 'Ge Hu'}))


def ground_cover(g):
    E, R, lon, lat, n1, n2 = g.ELEV_CELL, g.RELIEF_CELL, g.CELL_LON, g.CELL_LAT, g.NOISE, g.NOISE2
    cover = np.full(E.shape, g.GRASS)
    p = np.full(E.shape, 0.01)
    hills = (E > 35) | (R > 20)
    plain = ~hills
    near_water = g.water_cell.copy()
    for _ in range(2):
        near_water = near_water | g.shift(near_water, 1, 0, False) | g.shift(near_water, -1, 0, False) | \
            g.shift(near_water, 0, 1, False) | g.shift(near_water, 0, -1, False)
    coast = near_water & plain & (lon > 121.2)
    bamboo = hills & (lat < 31.45)                               # 宜溧山区 bamboo and tea
    ningzhen = hills & (lat >= 31.45)                            # 宁镇山脉, 茅山

    cover[plain & (n1 > 0.32)] = g.THICK                         # rice paddies and mulberry fields
    cover[coast & (n1 > 0.35)] = g.SAND                          # tidal flats of Chongming and the Pudong coast
    cover[hills & (E < 100) & (n2 > 0.55)] = g.THICK             # tea gardens
    p[plain] = 0.022                                             # village groves
    p[near_water & plain] = 0.05                                 # trees along the lake shores and canals
    p[hills] = 0.08
    p[bamboo] = 0.2 + 0.3 * (1 - n1[bamboo])
    p[ningzhen] = 0.12 + 0.2 * (1 - n1[ningzhen])
    p[coast] = 0.004
    return cover, hills, p


# --- buildings ---------------------------------------------------------------------------
from regions.pools import POOLS as CIVILIAN_POOLS   # noqa: E402

CIVILIAN_START_CLEAR = 7
FLAT_CITIES = [('上海', 121.47, 31.23, 6), ('南京', 118.78, 32.05, 3)]
PROP_DENSITY = 0.3
CIVILIAN_ZONES = [  # lon, lat, radius (cells), pool, density
    (121.47, 31.23, 4, 'highrise', 1.0),         # 浦西
    (121.56, 31.22, 3.5, 'highrise', 0.9),       # 浦东 陆家嘴 / 张江
    (121.43, 31.32, 3, 'suburb', 0.6),           # 宝山
    (121.60, 31.33, 3, 'industrial', 0.7),       # 外高桥
    (121.33, 31.18, 3, 'suburb', 0.6),           # 虹桥 / 闵行
    (118.78, 32.05, 3.5, 'highrise', 0.9),       # 南京 新街口
    (118.88, 32.12, 2.5, 'industrial', 0.6),     # 栖霞
    (120.75, 31.30, 3, 'industrial', 0.7),       # 苏州工业园区
    (120.98, 31.38, 3, 'industrial', 0.6),       # 昆山
    (120.89, 32.01, 3, 'suburb', 0.5),           # 南通
    (119.42, 32.38, 2.5, 'suburb', 0.5),         # 扬州
    (120.28, 31.88, 2.5, 'industrial', 0.6),     # 江阴
    (120.55, 31.87, 2.5, 'industrial', 0.6),     # 张家港
    (120.75, 31.65, 2.5, 'suburb', 0.5),         # 常熟
    (120.09, 30.89, 2.5, 'suburb', 0.5),         # 湖州
    (120.76, 30.77, 2.5, 'suburb', 0.5),         # 嘉兴
    (120.88, 31.12, 3, 'village', 0.45),         # 周庄 / 同里 water towns
    (121.05, 31.10, 3, 'village', 0.45),         # 朱家角
    (121.50, 31.62, 5, 'farm', 0.25),            # 崇明
    (120.40, 32.20, 5, 'farm', 0.25),            # 泰兴 / 如皋
    (119.75, 32.30, 5, 'farm', 0.25),            # 江都
    (119.25, 31.55, 5, 'farm', 0.25),            # 溧阳 / 金坛 plain
    (120.25, 31.20, 4, 'village', 0.35),         # 太湖 south shore
    (121.00, 31.60, 5, 'farm', 0.25),
    (121.60, 30.95, 4, 'farm', 0.25),            # 奉贤 / 南汇
]
CIVILIAN_FILL = ('small', 0.6)
