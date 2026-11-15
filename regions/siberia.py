"""反攻西伯利亚 (Siberia Counteroffensive, a what-if): the Chinese armies of the 三江平原 cross the 黑龙江 (Amur) and the
乌苏里江 (Ussuri) into the Russian Far East - the land north of the Amur and east of the Ussuri that the treaties of
Aigun (1858) and Peking (1860) gave to Russia, and that Chinese still loosely call Siberia. The map runs from 佳木斯 and
萝北 in the west to 伯力 (Khabarovsk) in the east, and from 比罗比詹 (Birobidzhan) in the north to 饶河 on the Ussuri.
The Amur, the Ussuri and the 松花江 are real water (OpenStreetMap centrelines), so the ground war is a fight for the
crossings: the 同江 rail bridge, the ports of 名山, 嘉荫, 抚远 and 饶河, the 乌苏大桥 and the Russian bridge that link
黑瞎子岛 (Bolshoy Ussuriysky island, split between the two countries in 2008) to either bank, and the Khabarovsk Amur
bridge, the only way between the 伯力 bank and the Jewish Autonomous Oblast. Early winter: the first snow is down and the
rivers have not frozen yet. Rivers, railways, bridges, airports and towns from OpenStreetMap (data/osm_siberia)."""
import json
import os

import numpy as np

from regions import osmwater

NAME = 'Siberia Counteroffensive'
OUTFILE = 'siberia.mpr'
SEED = 1860  # 中俄北京条约
W, H = 200, 153
LX, LY, LW, LH = 2, 4, 196, 140
LON0, LON1, LAT0, LAT1 = 130.22, 135.42, 49.05, 46.55
DEM_DIR, DEM_ZOOM = 'data/dem_siberia', 9
BLUR_SMOOTH, BLUR_WIDE, BLUR_TERRAIN = 1, 5, 2
ELEV_PTS = [0, 75, 120, 180, 260, 360, 480, 620, 800, 1100]   # the Amur 30-40 m, the 三江 plain 50-70 m, 小兴安岭 and the
LVL_PTS = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8]                      # JAO ridges 600-1000 m, 完达山 and Хехцир 800-950 m
THEATER = 'SNOW'
LIGHTING = dict(Red=0.93, Green=0.97, Blue=1.08, Level=0.02, Ambient=0.95)   # low November sun at 48°N

STARTS = [  # 中国 / 俄罗斯
    ('萝北', 130.825, 47.572),            # between the 松花江 and the Amur, opposite Амурзет
    ('富锦', 132.040, 47.165),            # the 三江 plain south of the 松花江
    ('饶河', 133.880, 46.830),            # on the Ussuri, opposite Покровка and Бикин
    ('抚远', 134.180, 48.200),            # the 东极 wedge between the Amur and the Ussuri
    ('比罗比詹 Биробиджан', 132.927, 48.793),   # capital of the Jewish Autonomous Oblast, on the Trans-Siberian
    ('列宁斯阔耶 Ленинское', 132.560, 48.040),  # behind the Russian end of the 同江 bridge
    ('伯力 Хабаровск', 135.270, 48.460),         # Far Eastern capital, on the high right bank of the Amur
    ('维亚泽姆斯基 Вяземский', 134.758, 47.533), # the Trans-Siberian south to Vladivostok, east of the Ussuri
]
START_FLAT, START_ORE_DIST, START_WATER_CLEAR = 9, 7.5, 3
MAP_CENTER = (132.9, 47.85)
ORE_TOWNS = [  # 三江 state farms and towns / Russian villages (positions from OSM)
    ('鹤北', 130.458, 47.539), ('宝泉岭', 130.535, 47.431), ('桦川', 130.713, 47.026), ('集贤', 131.125, 46.760),
    ('红兴隆', 131.568, 46.712), ('友谊', 131.806, 46.774), ('绥滨', 131.848, 47.286), ('普阳农场', 131.357, 47.278),
    ('二九〇农场', 132.100, 47.575), ('建三江', 132.634, 47.254), ('兴隆岗', 132.566, 46.972), ('大兴农场', 132.871, 46.981),
    ('青龙山农场', 132.958, 47.724), ('勤得利', 133.155, 47.972), ('前进农场', 133.125, 47.553), ('创业农场', 133.057, 47.365),
    ('红卫农场', 133.445, 47.329), ('洪河农场', 133.510, 47.587), ('鸭绿河农场', 133.650, 47.890), ('胜利农场', 133.867, 47.367),
    ('大佳河', 133.568, 47.184), ('山里', 133.418, 47.124), ('西林子', 134.001, 46.922), ('八五九农场', 134.047, 47.425),
    ('二道河农场', 134.153, 47.710), ('前哨农场', 134.124, 47.996), ('别拉洪', 134.376, 47.925), ('浓桥', 134.285, 48.189),
    ('保兴', 130.452, 48.600), ('红旗岭农场', 133.240, 46.837),
    ('Пузино', 131.211, 47.734), ('Благословенное', 131.382, 47.767), ('Нагибово', 131.532, 47.744),
    ('Биджан', 131.942, 47.975), ('Бабстово', 132.475, 48.121), ('Бирофельд', 132.663, 48.459),
    ('Валдгейм', 132.970, 48.696), ('Смидович', 133.814, 48.596), ('Николаевка', 134.779, 48.556),
    ('Екатерино-Никольское', 130.963, 47.752), ('Хор', 134.961, 47.883), ('Переяславка', 135.053, 47.967),
    ('Корфовский', 135.059, 48.223), ('Некрасовка', 135.232, 48.352), ('Бикин', 134.258, 46.819),
    ('Лермонтовка', 134.340, 47.149), ('Светлогорье', 134.450, 46.861)]
GEM_TOWNS = [('黑瞎子岛', 134.920, 48.400),             # Bolshoy Ussuriysky island, between the two bridges
             ('萝北云山石墨矿', 130.650, 47.720),         # the largest graphite field in Asia
             ('太平沟金矿', 130.560, 48.170),             # 观音山 placer gold on the Amur
             ('双鸭山煤矿 (岭东)', 131.250, 46.600),
             ('沃洛恰耶夫卡 Волочаевка', 134.560, 48.560)]  # Июнь-Корань hill, the 1922 battle of Volochaevka
OIL = []
TECH = [('CAAIRP', '佳木斯东郊机场', 130.460, 46.843, 3, 3),
        ('CAAIRP', '建三江湿地机场', 132.660, 47.107, 3, 3),
        ('CAAIRP', '伯力新机场 Хабаровск-Новый', 135.189, 48.528, 3, 3),
        ('CAAIRP', '比罗比詹南机场', 133.037, 48.578, 3, 3),
        ('CAOUTP', '同江口岸', 132.560, 47.700, 4, 3),
        ('CAOUTP', 'Нижнеленинское', 132.660, 48.000, 4, 3)]
BRIDGES = [  # crossings of the border rivers (OSM bridges; the ports get pontoon bridges)
    ('嘉荫-Пашково', 130.520, 48.880),
    ('名山-Амурзет', 131.080, 47.688),
    ('同江铁路大桥', 132.690, 47.900),
    ('抚远港', 134.300, 48.370),
    ('乌苏大桥 (黑瞎子岛)', 134.681, 48.284),
    ('黑瞎子岛-Бычиха', 134.900, 48.320),
    ('伯力阿穆尔大桥', 134.995, 48.535),
    ('饶河-Покровка', 134.040, 46.760),
    ('富绥松花江大桥', 131.968, 47.250),             # 松花江
]
CITIES = [('伯力', 135.090, 48.470, 3.5), ('佳木斯', 130.380, 46.795, 2.5), ('比罗比詹', 132.927, 48.793, 1.8),
          ('富锦', 132.025, 47.240, 1.4), ('同江', 132.504, 47.645, 1.2), ('抚远', 134.289, 48.356, 1.1),
          ('饶河', 134.008, 46.798, 1.0), ('萝北', 130.825, 47.572, 1.0), ('维亚泽姆斯基', 134.758, 47.533, 0.9),
          ('双鸭山', 131.140, 46.640, 1.6), ('建三江', 132.634, 47.254, 1.0), ('比金', 134.258, 46.819, 1.0)]

_D = 'data/osm_siberia'
_WATER = f'{_D}/water.json'


def _rail(names, guide, tol=0.06, step=1.5):
    """A line from the OSM rail ways named `names`: the points within `tol` degrees of a hand-drawn guide, ordered by
    their place along the guide and averaged every `step` km (double track and station yards fold into one bed)."""
    path = os.path.join(osmwater.HERE, _D, 'rail.json')
    P = np.array([(p['lon'], p['lat']) for e in json.load(open(path))['elements']
                  if e['tags'].get('name') in names and not e['tags'].get('service') for p in e['geometry']])
    k = np.array([0.67, 1.0])                         # degrees -> equal-km axes at 48°N
    g = np.array(guide) * k
    Q = P * k
    best_d, best_s = np.full(len(Q), np.inf), np.zeros(len(Q))
    s0 = 0.0
    for a, b in zip(g, g[1:]):
        L = np.linalg.norm(b - a)
        t = np.clip(np.einsum('ij,j->i', Q - a, b - a) / L ** 2, 0, 1)
        d = np.linalg.norm(Q - (a + t[:, None] * (b - a)), axis=1)
        m = d < best_d
        best_d[m], best_s[m] = d[m], s0 + t[m] * L
        s0 += L
    keep = best_d < tol
    Q, s = Q[keep], best_s[keep] * 111.0
    out = []
    for x in np.arange(0, s0 * 111.0 + step, step):
        m = (s >= x) & (s < x + step)
        if m.any():
            lon, lat = Q[m].mean(0) / k
            out.append((round(float(lon), 4), round(float(lat), 4)))
    return out


_TRANSSIB = 'Транссибирская магистраль'
RAILWAYS = {
    'Транссибирская магистраль': _rail({_TRANSSIB}, [
        (131.06, 49.02), (131.50, 49.02), (131.90, 49.01), (132.47, 49.00), (132.93, 48.79), (133.40, 48.65),
        (133.81, 48.60), (134.58, 48.56), (134.98, 48.53), (135.08, 48.50), (135.10, 48.40), (135.06, 48.22),
        (135.05, 47.97), (134.96, 47.88), (134.76, 47.53), (134.34, 47.15), (134.26, 46.82), (134.15, 46.55)]),
    '佳富线-福前线-前抚线': _rail({'佳富线', '福前线', '前抚线'}, [
        (130.43, 46.80), (130.80, 46.80), (131.13, 46.72), (131.75, 46.72), (131.90, 46.90), (132.00, 47.20),
        (132.40, 47.30), (132.90, 47.32), (133.12, 47.50), (133.20, 47.60), (134.05, 47.62), (134.12, 48.00),
        (134.30, 48.15), (134.45, 48.28), (134.51, 48.33)]),
    '同江线-同江铁路大桥-Ленинск': _rail({'同江线', '同江宽轨线', 'Унгун - Ленинск', 'Бирофельд - Унгун',
                                        'Биробиджан-2 - Бирофельд'}, [
        (132.34, 47.31), (132.42, 47.45), (132.50, 47.62), (132.63, 47.74), (132.69, 47.85), (132.70, 47.97),
        (132.60, 48.02), (132.45, 48.13), (132.37, 48.20), (132.45, 48.26), (132.66, 48.46), (132.89, 48.75)]),
}
P = {  # road nodes (OSM place positions)
    '佳木斯': (130.375, 46.800), '集贤': (131.125, 46.724), '富锦': (132.025, 47.240), '同江': (132.504, 47.640),
    '向阳川': (132.339, 47.292), '建三江': (132.634, 47.254), '前进': (133.125, 47.553), '抚远': (134.289, 48.356),
    '乌苏镇': (134.647, 48.250), '饶河': (134.008, 46.798), '小佳河': (133.731, 47.210), '萝北': (130.825, 47.572),
    '名山': (131.070, 47.678), '绥滨': (131.848, 47.286), '宝泉岭': (130.535, 47.431), '鹤岗': (130.300, 47.330),
    '嘉荫': (130.395, 48.886), '太平沟': (130.659, 48.131), '保兴': (130.452, 48.600), '二道河': (134.153, 47.710),
    '红兴隆': (131.568, 46.712), '友谊': (131.806, 46.774), '双鸭山': (131.140, 46.640), '大佳河': (133.568, 47.184),
    'Облучье': (131.060, 49.020), 'Бира': (132.466, 49.002), 'Биробиджан': (132.927, 48.793),
    'Смидович': (133.814, 48.596), 'Волочаевка': (134.578, 48.563), 'Хабаровск': (135.076, 48.481),
    'Корфовский': (135.059, 48.223), 'Переяславка': (135.053, 47.967), 'Вяземский': (134.758, 47.533),
    'Бикин': (134.258, 46.819), 'Ленинское': (132.622, 47.933), 'Бабстово': (132.475, 48.121),
    'Бирофельд': (132.663, 48.459), 'Амурзет': (131.090, 47.695), 'Пашково': (130.651, 48.886),
    'Биджан': (131.942, 47.975), 'Нижнеленинское': (132.660, 47.985), 'Бычиха': (134.825, 48.302),
    'Казакевичево': (134.736, 48.270), '黑瞎子岛': (134.760, 48.330), 'Покровка': (134.025, 46.716),
}


def _road(*names):
    return [P[n] for n in names]


TRACKS = {
    'G1011 哈同高速': _road('佳木斯', '集贤', '富锦', '向阳川', '同江'),
    'G1012 建黑高速': _road('建三江', '前进', '二道河', '抚远', '乌苏镇', '黑瞎子岛'),
    '建三江-富锦': _road('富锦', '建三江'),
    '饶河-建三江': _road('饶河', '大佳河', '小佳河', '建三江'),
    '饶河-双鸭山': _road('饶河', '红兴隆', '双鸭山'),
    '佳木斯-双鸭山': _road('佳木斯', '双鸭山'),
    '友谊-富锦': _road('红兴隆', '友谊', '富锦'),
    'S308 富锦-绥滨': _road('富锦', '绥滨'),
    '萝北-名山': _road('鹤岗', '宝泉岭', '萝北', '名山'),
    '萝北-绥滨': _road('萝北', '绥滨'),
    '嘉荫-太平沟-萝北': _road('嘉荫', '保兴', '太平沟', '萝北'),
    'Р-297 Амур': _road('Облучье', 'Бира', 'Биробиджан', 'Смидович', 'Волочаевка', 'Хабаровск'),
    'А-370 Уссури': _road('Хабаровск', 'Корфовский', 'Переяславка', 'Вяземский', 'Бикин'),
    'Биробиджан-Ленинское': _road('Биробиджан', 'Бирофельд', 'Бабстово', 'Ленинское', 'Нижнеленинское'),
    'Ленинское-Амурзет': _road('Ленинское', 'Биджан', 'Амурзет'),
    'Амурзет-Биробиджан': _road('Амурзет', 'Биджан', 'Бирофельд'),
    'Пашково-Облучье': _road('Пашково', 'Облучье'),
    'Бычиха-Казакевичево': _road('Хабаровск', 'Бычиха', 'Казакевичево'),
    'Бикин-Покровка': _road('Бикин', 'Покровка'),
}
TRACK_COVER = 'PAVE'
ROADS = dict(RAILWAYS, **TRACKS)
RIVER_NAMES = set()
RIVER_BBOX = (130.0, 46.3, 135.8, 49.3)
EXTRA_RIVERS = osmwater.lines(_WATER, {'Бира', 'Биджан', 'Хор', 'Бикин', 'Тунгуска', 'Кур', 'Урми', 'Самара',
                                       '挠力河', '别拉洪河', '梧桐河', '七星河', '嘉荫河', '浓江', '安邦河', '莲花河'})


def _spread(lines, dlat, lon0=134.56, lon1=135.03, lat0=48.22, lat1=48.50):
    """Move the channels round 黑瞎子岛 apart by up to `dlat` degrees (tapered to nothing at the island's ends): they are
    only 5-10 km apart, and at 1.4 km a cell the shore blocks of the two banks would meet over the island."""
    out = []
    for line in lines:
        out.append([(x, y + dlat * np.sin(np.pi * (x - lon0) / (lon1 - lon0)) if lon0 < x < lon1 and lat0 < y < lat1 else y)
                    for x, y in line])
    return out


def _ways(ids):
    """Polylines of the OSM water ways with these ids (the side channels share the river's name)."""
    return [[(p['lon'], p['lat']) for p in e['geometry']]
            for e in json.load(open(os.path.join(osmwater.HERE, _WATER)))['elements'] if e['id'] in ids]


# the border channel is tagged '黑龙江 Амур' except from the 松花江 mouth to 勤得利 (217386516) and from 黑瞎子岛 down past 伯力,
# where it is 'Амур' like the Russian side channels left out here; '黑龙江' 367888945 is the 抚远水道
_AMUR = _spread(osmwater.lines(_WATER, {'黑龙江 Амур'}) + _ways({217386516, 217386510, 808421134, 703523817, 703523818,
                                                                26927827}) + [[(132.511, 47.715), (132.549, 47.714)]],
                0.028) + _spread(_ways({367888945}), -0.028)
_NARROW = osmwater.lines(_WATER, {'Уссури / 乌苏里江', '乌苏里江', '松花江'}) + \
    _spread(osmwater.lines(_WATER, {'Амурская протока'}), -0.028)
WATER_RIVERS = _AMUR + _NARROW
WATER_HALF_WIDTH = 0.5     # the stamped centreline: the Ussuri and the 松花江 are 300-800 m wide
AMUR_HALF_WIDTH = 0.9      # the Amur 1-2 km
ISLANDS = [(lon, lat, 2.6) for lon, lat in (   # 黑瞎子岛 (Bolshoy Ussuriysky): kept as land between its channels
    (134.62, 48.360), (134.66, 48.352), (134.70, 48.345), (134.74, 48.335), (134.78, 48.330),
    (134.82, 48.350), (134.86, 48.370), (134.90, 48.390), (134.94, 48.400), (134.98, 48.410))]
LAKES = []
WATER_LEVEL = 0
MIN_WATER_CORNERS, MIN_LAND_CORNERS = 6, 8
BRIDGE_MAX_LEN = 16       # a longer deck would jump 黑瞎子岛 and both its channels at once
MOUNTAIN_TREES = ['TREE25', 'TREE26', 'TREE27', 'TREE05', 'TREE06', 'TREE07', 'TREE08', 'TREE15']   # larch, spruce
VALLEY_TREES = ['TREE01', 'TREE02', 'TREE03', 'TREE04', 'TREE10', 'TREE11', 'TREE12', 'TREE17']    # birch, willow
CHECK_POINTS = [(n, lon, lat) for n, lon, lat in GEM_TOWNS]


def water_corners(g):
    """The Amur a little wider than the stamped centreline, except round 黑瞎子岛, where its channels are only 5-10 km
    apart and a wider bed would leave the island a sliver."""
    island = (g.CORNER_LON > 134.55) & (g.CORNER_LON < 135.03) & (g.CORNER_LAT > 48.26) & (g.CORNER_LAT < 48.46)
    return (g.polyline_distance(_AMUR, offset=0.0) < AMUR_HALF_WIDTH) & ~island


def ground_cover(g):
    E, R, lon, lat, n1, n2 = g.ELEV_CELL, g.RELIEF_CELL, g.CELL_LON, g.CELL_LAT, g.NOISE, g.NOISE2
    cover = np.full(E.shape, g.GRASS)
    hills = (E > 140) | (R > 25)
    taiga = hills & ((E > 200) | (n1 < 0.6))
    near_river = (g.RIVER_D < 1.5) | g.water_cell
    for _ in range(2):
        near_river = near_river | g.shift(near_river, 1, 0, False) | g.shift(near_river, -1, 0, False) | \
            g.shift(near_river, 0, 1, False) | g.shift(near_river, 0, -1, False)
    plain = ~hills
    marsh = plain & near_river & (n2 > 0.55)                      # 三江 / Bira lowland wetlands: reeds through the snow
    fields = plain & ~marsh & (n1 > 0.66)                         # wind-scoured stubble; the rest of the farms lie under snow
    cover[fields] = g.THICK
    cover[marsh] = g.THICK
    cover[hills & (R > 45) & (n2 > 0.6)] = g.ROUGH                # rock on the ridges
    p = np.full(E.shape, 0.012)
    p[fields] = 0.004                                             # open prairie farms, shelter belts only
    p[marsh] = 0.05                                               # birch and willow on the river flats
    p[hills] = 0.14
    p[taiga] = 0.26 + 0.10 * (1 - n2[taiga])                      # larch and spruce taiga, open enough to march through
    return cover, hills, p


# --- buildings ---------------------------------------------------------------------------
from regions.pools import POOLS as CIVILIAN_POOLS   # noqa: E402

CIVILIAN_START_CLEAR = 5
FLAT_CITIES = [('佳木斯', 130.400, 46.775, 3), ('双鸭山', 131.120, 46.680, 3)]
PROP_DENSITY = 0.3
CIVILIAN_ZONES = [  # lon, lat, radius (cells), pool, density
    (130.420, 46.760, 3.5, 'suburb', 0.6),      # 佳木斯
    (131.120, 46.680, 3.5, 'industrial', 0.6),  # 双鸭山 coal town
    (132.20, 47.10, 6, 'farm', 0.2),            # 三江 state farms
    (133.10, 47.20, 6, 'farm', 0.2),
    (133.40, 47.60, 6, 'farm', 0.2),
    (131.40, 47.30, 6, 'farm', 0.2),
    (131.80, 46.90, 5, 'farm', 0.2),
    (133.90, 47.80, 5, 'farm', 0.2),
    (132.30, 48.35, 5, 'village', 0.2),         # JAO villages
    (133.70, 48.50, 5, 'village', 0.2),
    (134.90, 47.80, 4, 'village', 0.25),
]
CIVILIAN_FILL = ('small', 0.6)
