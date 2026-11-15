"""Battle of Shanghai (淞沪会战, 13 Aug - 26 Nov 1937): the street fighting in 闸北 and 虹口, the Japanese landings at
吴淞 and 川沙口, the "meat grinder" at 罗店, the 蕰藻浜 and 大场 battles, the retreat over the 苏州河 (四行仓库) and the
10th Army's landing at 金山卫 that turned the whole line. The map runs from 昆山 and 太仓 to the mouth of the 长江 and
from 浏河 down to the 杭州湾. The 黄浦江 (with its upper course to 淀山湖), the 苏州河/吴淞江 and the lower 蕰藻浜 are
real water with only a few crossings. Rivers, lakes, the 1908 沪宁 and 1909 沪杭 railways and the towns come from
OpenStreetMap (data/osm_songhu: water.json and rail.json are read at run time)."""
import json
import os

import numpy as np

from regions import osmwater

NAME = 'Battle of Shanghai 1937'
OUTFILE = 'songhu.mpr'
SEED = 1937  # 八一三
W, H = 196, 204
LX, LY, LW, LH = 2, 4, 192, 191
LON0, LON1, LAT0, LAT1 = 120.88, 121.98, 31.62, 30.68
DEM_DIR, DEM_ZOOM = 'data/dem_songhu', 10
BLUR_SMOOTH, BLUR_WIDE, BLUR_TERRAIN = 1, 5, 2
ELEV_PTS = [0, 12, 30, 60, 100, 150]   # the delta is 2-5 m above the sea; only the 佘山 hills stand out
LVL_PTS = [0, 0, 1, 2, 3, 4]
LIGHTING = dict(Red=1.02, Green=0.98, Blue=0.93, Level=0.02, Ambient=0.96)   # autumn haze and the smoke of 闸北

STARTS = [  # 中国军队 / 日军
    ('闸北·江湾', 121.478, 31.300),     # 八十八师 at 八字桥 and 江湾; the city to the south is neutral ground
    ('罗店', 121.351, 31.397),          # "血肉磨坊": the crossroads behind the 川沙口 and 浏河 beaches
    ('浦东', 121.600, 31.180),          # 张发奎's right wing army and its guns on the far bank of the 黄浦江
    ('太仓', 121.126, 31.459),          # the left wing's rear on the 浏河, where the Japanese landed again on 13 Nov
    ('昆山', 120.976, 31.387),          # the 沪宁 railway, the road back to 苏州 and the 吴福线
    ('青浦', 121.119, 31.152),          # the 淀山湖 gap on the retreat west
    ('松江', 121.223, 31.034),          # held for three days by 67 军 after the 金山卫 landing
    ('金山卫', 121.300, 30.765),        # Japanese 10th Army lands on 5 Nov 1937
]
START_FLAT, START_ORE_DIST, START_WATER_CLEAR = 8, 7.0, 2
MAP_CENTER = (121.35, 31.17)
ORE_TOWNS = [  # market towns of the delta (positions from OSM)
    ('嘉定', 121.261, 31.378), ('南翔', 121.316, 31.296), ('外冈', 121.158, 31.365), ('安亭', 121.160, 31.296),
    ('沙溪', 121.066, 31.559), ('娄塘', 121.202, 31.420), ('千灯', 121.020, 31.257), ('嘉善', 120.930, 30.840),
    ('枫泾', 121.010, 30.891), ('新埭', 121.077, 30.822), ('朱家角', 121.042, 31.106), ('泗泾', 121.267, 31.121),
    ('七宝', 121.346, 31.160), ('江桥', 121.327, 31.241), ('莘庄', 121.373, 31.112), ('叶榭', 121.316, 30.946),
    ('亭林', 121.309, 30.885), ('朱泾', 121.162, 30.901), ('漕泾', 121.406, 30.796), ('南桥', 121.454, 30.906),
    ('奉城', 121.642, 30.916), ('南汇', 121.745, 31.050), ('周浦', 121.571, 31.116), ('川沙', 121.700, 31.195),
    ('高桥', 121.571, 31.344), ('长兴岛', 121.726, 31.389), ('月浦', 121.417, 31.422), ('平湖', 121.012, 30.712), ('石湖荡', 121.115, 30.987)]
GEM_TOWNS = [('大场', 121.412, 31.300),        # fell 26 Oct: the whole 闸北-江湾 line had to go back over the 苏州河
             ('宝山城', 121.485, 31.405),      # 姚子青营, 600 men, held the walled town to the last
             ('浏河', 121.272, 31.517),        # the 川沙口 and 浏河 landing beaches
             ('佘山', 121.190, 31.100),        # the only hills on the Shanghai plain
             ('崇明岛', 121.746, 31.549)]      # across the 长江 south channel: only by sea or air
OIL = []
TECH = [('CAAIRP', '虹桥机场', 121.340, 31.195, 3, 3),     # the 9 Aug 虹桥 incident that lit the fuse
        ('CAAIRP', '龙华机场', 121.450, 31.172, 3, 3),
        ('CAHOSP', '红十字会医院 (徐家汇)', 121.425, 31.190, 6, 4),
        ('CAOUTP', '吴淞炮台', 121.500, 31.385, 4, 3),
        ('CAOUTP', '狮子林炮台', 121.425, 31.440, 4, 3)]
LANDMARKS = [('CANEWY20', '四行仓库', 121.455, 31.252, 3, 5)]       # "八百壮士", 26 Oct - 1 Nov
BRIDGES = [  # bridges and ferries of 1937
    ('西藏路桥 (四行仓库)', 121.474, 31.243),   # 苏州河, east to west
    ('梵王渡', 121.418, 31.226),
    ('北新泾', 121.370, 31.218),
    ('黄渡', 121.210, 31.266),
    ('赵屯', 121.110, 31.282),
    ('陆家浜', 121.030, 31.290),
    ('吴淞镇', 121.488, 31.371),                 # 蕰藻浜
    ('陈家行', 121.400, 31.338),
    ('董家渡', 121.505, 31.215),                 # 黄浦江 and its upper course, downstream to upstream
    ('杨树浦渡', 121.545, 31.270),
    ('龙华渡', 121.478, 31.170),
    ('闵行渡', 121.420, 31.015),
    ('米市渡', 121.250, 30.965),
    ('石湖荡铁桥', 121.150, 30.972),
    ('泖河渡', 121.050, 31.040),
]
CITIES = [('上海', 121.475, 31.240, 10), ('松江', 121.226, 31.024, 1.6), ('昆山', 120.976, 31.387, 1.6),
          ('嘉定', 121.258, 31.380, 1.5), ('太仓', 121.126, 31.459, 1.5), ('青浦', 121.119, 31.152, 1.3),
          ('吴淞', 121.490, 31.378, 1.1), ('南翔', 121.316, 31.296, 1.0)]


def _rail(name, guide, step=0.02, tol=0.02):
    """A trunk line from the OSM ways of `name`: the points near a hand-drawn guide, averaged in steps of longitude."""
    path = os.path.join(osmwater.HERE, 'data/osm_songhu/rail.json')
    P = np.array([(p['lon'], p['lat']) for e in json.load(open(path))['elements']
                  if e['tags'].get('name') == name for p in e['geometry']])
    g = np.array(guide)
    d = np.full(len(P), np.inf)
    for a, b in zip(g, g[1:]):
        t = np.clip(np.einsum('ij,j->i', P - a, b - a) / np.dot(b - a, b - a), 0, 1)
        d = np.minimum(d, np.hypot(*(P - (a + t[:, None] * (b - a))).T))
    P = P[d < tol]
    out = []
    for x in np.arange(g[:, 0].min(), g[:, 0].max() + step, step):
        m = (P[:, 0] >= x) & (P[:, 0] < x + step)
        if m.any():
            out.append((round(float(P[m, 0].mean()), 4), round(float(P[m, 1].mean()), 4)))
    return out


RAILWAYS = {
    '沪宁铁路 (1908)': _rail('京沪线', [(120.86, 31.375), (120.95, 31.385),
                                       (121.15, 31.30), (121.32, 31.30), (121.40, 31.255), (121.455, 31.255)])
    + [(121.475, 31.253)],                                                   # 上海北站, 宝山路
    '沪杭铁路 (1909)': _rail('沪昆线', [(120.86, 30.83), (120.92, 30.845), (121.01, 30.89),
                                       (121.23, 31.00), (121.38, 31.11), (121.45, 31.16)])
    + [(121.42, 31.135), (121.445, 31.165), (121.465, 31.19), (121.48, 31.205)],   # 新龙华 - 龙华 - 上海南站
    '淞沪铁路': [(121.475, 31.253), (121.478, 31.275), (121.482, 31.300), (121.488, 31.330), (121.493, 31.360),
                 (121.497, 31.375), (121.503, 31.392)],                      # 北站 - 江湾 - 吴淞 - 炮台湾
}
TRACKS = {  # highways of 1937, gravel and dirt
    '沪太公路': [(121.440, 31.282), (121.412, 31.289), (121.385, 31.33), (121.368, 31.359), (121.351, 31.397),
                 (121.31, 31.46), (121.272, 31.517), (121.20, 31.49), (121.126, 31.459)],
    '宝山-罗店-嘉定': [(121.49, 31.378), (121.485, 31.407), (121.417, 31.422), (121.351, 31.397), (121.30, 31.39),
                       (121.261, 31.378)],
    '沪嘉公路': [(121.425, 31.258), (121.398, 31.252), (121.316, 31.296), (121.28, 31.34), (121.261, 31.378),
                 (121.20, 31.42), (121.126, 31.459)],
    '昆太公路': [(120.976, 31.390), (121.05, 31.43), (121.126, 31.455)],
    '沪青公路': [(121.405, 31.205), (121.34, 31.20), (121.25, 31.18), (121.119, 31.152), (121.042, 31.106)],
    '松青公路': [(121.223, 31.040), (121.17, 31.10), (121.119, 31.152), (121.06, 31.21), (121.02, 31.257),
                 (121.00, 31.33), (120.976, 31.38)],
    '松金公路': [(121.223, 31.030), (121.25, 30.975), (121.316, 30.946), (121.309, 30.885), (121.294, 30.804),
                 (121.300, 30.770)],
    '沪闵公路': [(121.425, 31.175), (121.40, 31.13), (121.373, 31.112), (121.40, 31.06), (121.42, 31.03)],
    '浦东公路': [(121.555, 31.235), (121.60, 31.20), (121.70, 31.19), (121.745, 31.05), (121.642, 30.916),
                 (121.454, 30.906), (121.406, 30.796), (121.300, 30.770)],
    '浦东-周浦': [(121.60, 31.18), (121.571, 31.116), (121.52, 31.02), (121.454, 30.906)],
    '平湖-金山卫': [(120.88, 30.735), (121.012, 30.712), (121.15, 30.74), (121.300, 30.765)],
}
TRACK_COVER = 'SAND'
ROADS = dict(RAILWAYS, **TRACKS)
WATER = 'data/osm_songhu/water.json'
RIVER_NAMES = set()
RIVER_BBOX = (120.4, 30.5, 122.2, 31.9)
EXTRA_RIVERS = osmwater.lines(WATER, {'浏河', '老浏河', '新浏河', '娄江', '练祁河', '杨林塘', '盐铁塘', '白茆塘', '走马塘',
                                      '横沥河', '大运河', '京杭大运河', '江南运河', '大泖港', '虬江', '西虬江', '千灯浦',
                                      '朱泖河', '北泖泾', '斜塘港', '浦东运河', '老蕰藻浜'})


def _water_rivers():
    """黄浦江 with its upper course (横潦泾 - 泖河 - 拦路港 to 淀山湖), 苏州河/吴淞江 from 太湖, and the 蕰藻浜 below 南翔
    (its upper reach was cut through to the 吴淞江 only in the 1950s)."""
    rivers = osmwater.lines(WATER, {'黄浦江', '横潦泾', '竖潦泾', '泖河', '拦路港', '苏州河', '吴淞江'})
    yzb = [[(x, y) for x, y in line if x >= 121.29] for line in osmwater.lines(WATER, {'蕰藻浜'})]
    return rivers + [line for line in yzb if len(line) > 1]


WATER_RIVERS = _water_rivers()
WATER_HALF_WIDTH = 0.5    # just the stamped centreline: the 苏州河 is 50 m wide, the 黄浦江 400 m
LAKES = []
WATER_LEVEL = 0
MIN_WATER_CORNERS, MIN_LAND_CORNERS = 6, 8
BRIDGE_DETOUR, BRIDGE_MAX_LEN, BRIDGE_SEEDS = 40, 14, 120   # the rivers end inland (蕰藻浜 at 南翔, 吴淞江 at 太湖): bridge them where the way round is long
LAKE_NAMES = {'太湖', '淀山湖', '阳澄西湖', '阳澄中湖', '阳澄东湖', '阳澄湖', '傀儡湖', '澄湖', '金鸡湖', '独墅湖', '昆承湖',
              '尚湖', '元荡', '汾湖'}
MOUNTAIN_TREES = ['TREE01', 'TREE02', 'TREE03', 'TREE04', 'TREE10', 'TREE11', 'TREE12', 'TREE17']
VALLEY_TREES = ['TREE01', 'TREE02', 'TREE03', 'TREE04', 'TREE10', 'TREE11', 'TREE12', 'TREE17']
FRONT_1937 = [  # the left wing's line from late August to 26 October: 浏河 - 罗店 - 刘行 - 蕰藻浜 - 大场 - 江湾 - 闸北
    [(121.255, 31.525), (121.30, 31.47), (121.33, 31.43), (121.351, 31.397), (121.368, 31.359), (121.395, 31.335),
     (121.412, 31.300), (121.445, 31.290), (121.475, 31.280), (121.500, 31.268)],
]
CHECK_POINTS = [('大场', 121.412, 31.300), ('宝山城', 121.485, 31.405), ('佘山', 121.190, 31.100),
                ('南汇', 121.745, 31.050), ('崇明岛', 121.746, 31.549), ('长兴岛', 121.726, 31.389)]


def water_corners(g):
    """The 长江 mouth and 杭州湾 from the bathymetry, the lakes from OSM."""
    sea = g.sample(g.box_blur(g.DEM, 1), g.CORNER_LON, g.CORNER_LAT) < -0.5
    return sea | osmwater.mask(g, osmwater.polygons(WATER, LAKE_NAMES))


def ground_cover(g):
    E, R, lon, lat, n1, n2 = g.ELEV_CELL, g.RELIEF_CELL, g.CELL_LON, g.CELL_LAT, g.NOISE, g.NOISE2
    cover = np.full(E.shape, g.GRASS)
    hills = (E > 25) | (R > 12)
    plain = ~hills
    near_water = g.water_cell.copy()
    for _ in range(2):
        near_water = near_water | g.shift(near_water, 1, 0, False) | g.shift(near_water, -1, 0, False) | \
            g.shift(near_water, 0, 1, False) | g.shift(near_water, 0, -1, False)
    coast = near_water & plain & ((lon > 121.40) | (lat < 30.80))      # mud flats of the 长江 mouth and 杭州湾
    cotton = plain & (((lon > 121.15) & (lat > 31.33)) | (lon > 121.55))   # 嘉定, 宝山 and 浦东 grew cotton
    canal = g.RIVER_D < 1.2
    front = g.polyline_distance(FRONT_1937) < 4.5 + 2.0 * (n2 - 0.5)

    cover[plain & (n1 > 0.30)] = g.THICK                       # rice paddies, stubble after the harvest
    cover[cotton & (n1 > 0.55)] = g.ROUGH                      # cotton fields
    cover[coast & (n1 > 0.35)] = g.SAND
    p = np.full(E.shape, 0.02)                                 # bamboo groves around every hamlet
    p[canal] = 0.08                                            # willows along the creeks and canals
    p[near_water & plain] = 0.05
    p[hills] = 0.25 + 0.2 * (1 - n1[hills])
    p[coast] = 0.004
    cover[front & (n1 > 0.25)] = g.ROUGH                       # shell craters
    cover[front & (n1 <= 0.25)] = g.SAND                       # trenches and churned mud
    p[front] = 0.006
    return cover, hills, p


# --- buildings ---------------------------------------------------------------------------
from regions.pools import POOLS   # noqa: E402

CIVILIAN_POOLS = dict(POOLS)
CIVILIAN_POOLS['bund'] = ['CAWASH03', 'CAWASH04', 'CAWASH05', 'CAWASH06', 'CAWASH07', 'CAWASH09', 'CAPARS04',
                          'CAPARS05', 'CAPARS06', 'CANEWY01', 'CANWY05', 'CACHIG04']      # 外滩 and 南京路
CIVILIAN_POOLS['lilong'] = ['CAHSE01', 'CAHSE02', 'CAHSE03', 'CAHSE04', 'CAHSE07', 'CAPARS10', 'CAPARS13', 'CAPARS14',
                            'CAEUR04']                                                  # 石库门 and 法租界 villas
CIVILIAN_START_CLEAR = 5
FLAT_CITIES = [('上海', 121.475, 31.240, 10)]
PROP_DENSITY = 0.3
CIVILIAN_ZONES = [  # lon, lat, radius (cells), pool, density; the villages on the 1937 front line stay empty
    (121.487, 31.237, 5, 'bund', 1.0),         # 外滩 and 南京路, 公共租界
    (121.450, 31.215, 5, 'lilong', 0.8),         # 法租界
    (121.497, 31.210, 4, 'lilong', 0.7),         # 南市
    (121.435, 31.240, 4, 'lilong', 0.6),         # 静安寺 / 曹家渡
    (121.497, 31.262, 4, 'suburb', 0.6),         # 虹口, the Japanese quarter
    (121.535, 31.268, 5, 'industrial', 0.7),   # 杨树浦 mills
    (121.525, 31.232, 3.5, 'industrial', 0.5),   # 浦东 wharves
    (121.258, 31.380, 7, 'village', 0.3),     # 嘉定
    (121.042, 31.106, 7, 'village', 0.3),     # 朱家角
    (121.316, 31.296, 7, 'village', 0.3),        # 南翔
    (121.010, 30.891, 7, 'village', 0.3),        # 枫泾
    (120.930, 30.840, 7, 'village', 0.3),        # 嘉善
    (121.000, 31.500, 6, 'farm', 0.2),           # 太仓 / 昆山 plain
    (121.000, 31.180, 5, 'farm', 0.2),           # 淀山湖 north
    (121.230, 31.200, 5, 'farm', 0.2),
    (121.050, 30.960, 6, 'farm', 0.2),           # 枫泾 / 嘉善
    (121.150, 30.800, 6, 'farm', 0.2),           # 金山
    (121.480, 30.960, 6, 'farm', 0.2),           # 奉贤
    (121.720, 31.090, 6, 'farm', 0.2),           # 南汇
    (121.620, 31.300, 5, 'farm', 0.2),           # 高桥
]
CIVILIAN_FILL = ('small', 0.6)
