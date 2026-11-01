"""Liaoshen Campaign (辽沈战役, Sep-Nov 1948): the Liaoxi corridor from Tashan to Jinzhou, the siege of Changchun,
the Heishan-Dahushan blocking line and Shenyang, with Liaodong Bay from the bathymetry."""
import numpy as np

NAME = 'Liaoshen Campaign'
OUTFILE = 'liaoshen.mpr'
SEED = 1948  # 9-12 Sep to 2 Nov 1948
W, H = 188, 186
LX, LY, LW, LH = 2, 4, 184, 173
LON0, LON1, LAT0, LAT1 = 120.40, 125.80, 44.15, 40.35
DEM_DIR, DEM_ZOOM = 'data/dem_liaoshen', 8
BLUR_SMOOTH, BLUR_WIDE, BLUR_TERRAIN = 1, 4, 1
ELEV_PTS = [0, 40, 90, 150, 300, 420, 540, 680, 850, 1100]
LVL_PTS = [0, 0, 1, 2, 2.4, 4, 5, 6, 7, 8.5]   # the Songliao divide (150-300 m) stays one plateau
THEATER = 'SNOW'   # the campaign ended on 2 Nov 1948; Manchuria already had its first snow
LIGHTING = dict(Red=0.98, Green=0.98, Blue=1.03, Level=0.03, Ambient=1.0)

STARTS = [  # the battle's eight fronts
    ('长春', 125.32, 43.88),       # 围困长春 (郑洞国)
    ('四平', 124.35, 43.17),
    ('沈阳', 123.43, 41.80),       # 卫立煌 / 沈阳总部
    ('营口', 122.50, 40.72),       # 第52军由营口海上撤退
    ('黑山', 122.116, 41.687),     # 黑山-大虎山阻击战, 廖耀湘兵团覆灭
    ('义县', 121.241, 41.535),     # 攻克义县, 锦州外围
    ('锦州', 121.12, 41.13),       # 攻克锦州 (范汉杰)
    ('锦西·塔山', 120.80, 40.80),   # 塔山阻击战: 东进兵团 from 锦西 and 葫芦岛
]
START_FLAT, START_ORE_DIST, START_WATER_CLEAR = 10, 7.5, 3
MAP_CENTER = (122.6, 42.1)
ORE_TOWNS = [('阜新', 121.67, 42.02), ('新民', 122.83, 41.99), ('彰武', 122.54, 42.38), ('铁岭', 123.84, 42.29),
             ('开原', 124.04, 42.55), ('法库', 123.40, 42.50), ('康平', 123.35, 42.74), ('昌图', 124.11, 42.79),
             ('公主岭', 124.82, 43.50), ('双辽', 123.50, 43.52), ('伊通', 125.30, 43.35), ('辽源', 125.14, 42.89),
             ('西丰', 124.72, 42.74), ('辽阳', 123.17, 41.27), ('海城', 122.75, 40.85), ('盘山', 122.05, 41.24),
             ('台安', 122.43, 41.41), ('北镇', 121.80, 41.60), ('兴城', 120.73, 40.61), ('北票', 120.77, 41.80),
             ('沟帮子', 121.97, 41.37), ('析木', 123.10, 40.78), ('清原', 124.92, 42.10), ('东丰', 125.53, 42.68),
             ('双阳', 125.66, 43.52), ('辽中', 122.72, 41.50), ('梨树', 124.33, 43.31), ('朝阳', 120.55, 41.57),
             ('库伦', 121.77, 42.73), ('通辽南', 122.60, 43.20), ('科左后旗', 122.36, 42.94), ('长岭', 123.98, 44.00),
             ('凌海', 121.36, 41.17), ('黑山北', 122.05, 41.95)]
GEM_TOWNS = [('鞍山铁矿', 122.95, 41.05), ('本溪湖煤铁', 123.77, 41.33), ('抚顺煤矿', 123.92, 41.86),
             ('阜新煤矿', 121.62, 41.95), ('医巫闾山', 121.68, 41.70), ('辽源煤矿', 125.16, 42.93)]
OIL = [('辽河油田', 122.07, 41.12), ('锦西炼油厂', 120.80, 40.73), ('抚顺石油一厂', 123.84, 41.90),
       ('盘锦北', 121.90, 41.20), ('法库东', 123.60, 42.45)]
TECH = [('CAAIRP', '阜新机场', 121.72, 42.12, 3, 3),
        ('CAHOSP', '新民', 122.83, 41.92, 6, 4),
        ('CAOUTP', '大虎山车站', 122.16, 41.55, 4, 3)]
CITIES = [('沈阳', 123.43, 41.80, 4), ('长春', 125.32, 43.88, 3.5), ('锦州', 121.13, 41.10, 2.5), ('鞍山', 122.99, 41.11, 2.2),
          ('抚顺', 123.95, 41.88, 2.2), ('营口', 122.23, 40.67, 2), ('四平', 124.35, 43.17, 2), ('本溪', 123.77, 41.30, 1.6),
          ('辽阳', 123.17, 41.27, 1.6), ('阜新', 121.67, 42.02, 1.8), ('铁岭', 123.84, 42.29, 1.4), ('葫芦岛', 120.84, 40.73, 1.4),
          ('黑山', 122.116, 41.687, 1.2), ('义县', 121.241, 41.535, 1.2), ('新民', 122.83, 41.99, 1.2)]
ROADS = {  # the railways of 1948
    '北宁铁路': [(123.43, 41.80), (122.83, 41.99), (122.147, 41.616), (121.97, 41.37), (121.13, 41.10), (120.933, 40.845),
             (120.84, 40.73), (120.73, 40.61), (120.40, 40.38)],
    '中长铁路': [(125.32, 43.88), (124.82, 43.50), (124.35, 43.17), (124.11, 42.79), (124.04, 42.55), (123.84, 42.29),
             (123.43, 41.80), (123.17, 41.27), (122.99, 41.11), (122.75, 40.85), (122.50, 40.63), (122.40, 40.35)],
    '大营支线': [(122.50, 40.63), (122.27, 40.66)],
    '沟营线': [(121.97, 41.37), (122.05, 41.12), (122.27, 40.66)],
    '锦承线': [(121.13, 41.10), (121.241, 41.535), (120.40, 41.57)],
    '大郑线': [(122.147, 41.616), (122.116, 41.687), (122.54, 42.38), (123.50, 43.52), (123.50, 44.15)],
    '新义线': [(121.241, 41.535), (121.67, 42.02), (122.40, 42.05)],
    '沈吉线': [(123.43, 41.80), (123.95, 41.88), (124.92, 42.10), (125.80, 42.20)],
    '四梅线': [(124.35, 43.17), (124.72, 42.74), (125.14, 42.89), (125.80, 42.55)],
    '平齐线': [(124.35, 43.17), (123.50, 43.52)],
}
RIVER_NAMES = {'Liao', 'Daling', 'Xiliao'}
RIVER_BBOX = (120.0, 40.0, 126.2, 44.5)
EXTRA_RIVERS = [  # traced by hand
    [(123.95, 41.88), (123.43, 41.76), (122.95, 41.60), (122.72, 41.45), (122.42, 41.02), (122.23, 40.70)],   # 浑河
    [(123.77, 41.30), (123.17, 41.26), (122.80, 41.15), (122.42, 41.02)],                                    # 太子河
    [(121.95, 42.30), (122.05, 41.85), (122.15, 41.45), (122.10, 41.10)],                                    # 绕阳河
    [(120.70, 41.45), (120.95, 41.20), (121.13, 41.07), (121.25, 40.88)],                                    # 小凌河
    [(120.62, 40.95), (120.80, 40.82), (120.95, 40.78)],                                                      # 女儿河
]
WATER_RIVERS, WATER_HALF_WIDTH, LAKES = [], 0, []
WATER_LEVEL = 0
MIN_WATER_CORNERS, MIN_LAND_CORNERS = 6, 8
MOUNTAIN_TREES = ['TREE25', 'TREE26', 'TREE27', 'TREE05', 'TREE06', 'TREE07', 'TREE08', 'TREE15']
VALLEY_TREES = ['TREE01', 'TREE02', 'TREE03', 'TREE04', 'TREE10', 'TREE11', 'TREE12', 'TREE17']


def water_corners(g):
    """Liaodong Bay from the bathymetry; the tidal ponds behind the Panjin reed flats are dropped."""
    sea = g.sample(g.box_blur(g.DEM, 1), g.CORNER_LON, g.CORNER_LAT) < -0.5
    lab, sizes = g.components(sea)
    return (lab >= 0) & (sizes[lab] > 3000)


def ground_cover(g):
    E, R, lon, lat, n1, n2 = g.ELEV_CELL, g.RELIEF_CELL, g.CELL_LON, g.CELL_LAT, g.NOISE, g.NOISE2
    cover = np.full(E.shape, g.GRASS)
    p = np.full(E.shape, 0.008)
    hills = (R > 30) | (E > 350)                                                   # relief, not height: the Songliao divide is farmland
    plain = ~hills
    near_water = g.water_cell.copy()
    for _ in range(3):
        near_water = near_water | g.shift(near_water, 1, 0, False) | g.shift(near_water, -1, 0, False) | \
            g.shift(near_water, 0, 1, False) | g.shift(near_water, 0, -1, False)
    reeds = near_water & plain & (lon > 121.6) & (lon < 122.6)                     # 盘锦 reed marsh
    east = (E > 260) & (lon > 123.2)                                               # 辽东 / 吉林 forested mountains
    west = hills & (lon < 122.3)                                                    # 辽西 bare loess hills
    riparian = (g.RIVER_D < 1.3) & plain

    cover[plain & (n1 > 0.5)] = g.THICK                       # first thin snow: sorghum stubble shows through
    cover[plain & (lat > 43.0) & (n2 > 0.6)] = g.THICK        # 科尔沁 steppe edge, wind-blown
    cover[west & (n2 > 0.45)] = g.ROUGH                       # snowy rock on the bare Liaoxi hills
    cover[reeds & (n1 > 0.25)] = g.SAND                       # frozen 盘锦 reed marsh
    p[plain] = 0.008
    p[riparian] = 0.08
    p[west] = 0.03
    p[east] = 0.1 + 0.2 * (1 - n1[east])
    p[reeds] = 0.002
    return cover, hills, p


# --- buildings ---------------------------------------------------------------------------
from regions.pools import POOLS as CIVILIAN_POOLS   # noqa: E402

CIVILIAN_START_CLEAR = 7
FLAT_CITIES = [('沈阳', 123.43, 41.80, 4)]
PROP_DENSITY = 0.3
CIVILIAN_ZONES = [  # lon, lat, radius (cells), pool, density
    (123.95, 41.88, 2.5, 'industrial', 0.7),     # 抚顺
    (122.99, 41.11, 2.5, 'industrial', 0.7),     # 鞍山 / 昭和制钢所
    (123.77, 41.30, 2, 'industrial', 0.6),       # 本溪
    (123.17, 41.27, 2, 'suburb', 0.5),           # 辽阳
    (121.67, 42.02, 2, 'industrial', 0.5),       # 阜新
    (123.84, 42.29, 2, 'suburb', 0.5),           # 铁岭
    (122.83, 41.99, 2, 'village', 0.5),          # 新民
    (124.10, 42.70, 5, 'farm', 0.25),
    (124.80, 43.40, 5, 'farm', 0.25),
    (123.00, 42.70, 5, 'farm', 0.25),
    (122.70, 41.50, 4, 'farm', 0.25),
    (121.70, 41.40, 4, 'village', 0.3),
    (121.00, 41.80, 4, 'village', 0.25),
    (125.00, 42.40, 4, 'village', 0.25),
]
CIVILIAN_FILL = ('small', 0.6)
