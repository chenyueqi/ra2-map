"""楚汉争霸 6: Battle of Gaixia (垓下之战, December 202 BC). 项羽, walled in at 垓下 by 韩信, 刘邦, 彭越 and 英布,
hears the songs of 楚 on all sides; he breaks out south with 800 riders, crosses the 淮水, is led astray into the great
marsh at 阴陵, fights his last stand at 东城 and dies at the 乌江 ferry on the 长江, refusing to cross to 江东.
The map runs from 垓下 down to the 长江; the 淮水 and the 长江 are real water with only a few crossings.
River courses and places from OpenStreetMap (data/osm_chuhan6_gaixia: water.json is read at run time)."""
import numpy as np

from regions import osmwater

NAME = 'Chu-Han 6: Battle of Gaixia'
OUTFILE = 'chuhan6_gaixia.mpr'
SEED = 202  # 202 BC
W, H = 164, 198
LX, LY, LW, LH = 2, 4, 160, 185
LON0, LON1, LAT0, LAT1 = 116.45, 118.70, 33.86, 31.66
DEM_DIR, DEM_ZOOM = 'data/dem_chuhan6_gaixia', 10
BLUR_SMOOTH, BLUR_WIDE, BLUR_TERRAIN = 1, 5, 2
ELEV_PTS = [0, 28, 45, 75, 120, 180, 250, 330, 450]   # Huai plain 15-25 m, Jianghuai divide 50-100 m, 琅琊/涂山 300+
LVL_PTS = [0, 0, 1, 2, 3, 4, 5, 6, 7]
THEATER = 'SNOW'   # the twelfth month: "四面皆楚歌" on a winter night
LIGHTING = dict(Red=0.93, Green=0.95, Blue=1.08, Level=0.02, Ambient=0.92)   # cold dusk

STARTS = [  # 项羽 at 垓下 / the five armies around him / his flight south
    ('垓下 (项羽)', 117.55, 33.29),          # 垓下遗址, 固镇 濠城 (estimated)
    ('灵璧 (刘邦)', 117.553, 33.543),        # 汉王 behind 韩信's army; 虞姬墓 nearby
    ('符离 (韩信)', 116.972, 33.754),        # 齐王韩信 with 30万, coming from 彭城
    ('蒙 (彭越)', 116.566, 33.258),          # 梁王彭越 from the west
    ('寿春 (英布/刘贾)', 116.778, 32.578),   # 九江 army from the south
    ('钟离', 117.62, 32.86),                 # the 淮水 crossing; 项羽 "渡淮"
    ('东城', 117.70, 32.30),                 # "东城快战", 四溃山 (estimated)
    ('乌江', 118.40, 31.88),                 # 乌江亭长舣船待, the ferry to 江东
]
START_FLAT, START_ORE_DIST, START_WATER_CLEAR = 9, 7.5, 3
MAP_CENTER = (117.55, 32.75)
ORE_TOWNS = [  # Han-era county seats where known, else today's market towns (positions from OSM)
    ('蕲 (大泽乡)', 117.091, 33.516), ('谷阳', 117.315, 33.322), ('洨', 116.896, 33.425), ('龙亢', 116.890, 33.136),
    ('虹', 117.656, 33.084), ('夏丘', 117.863, 33.468), ('沱湖', 117.834, 33.157), ('徐', 118.413, 33.236),
    ('顺山集', 118.168, 33.579), ('下蔡', 116.706, 32.712), ('当涂 (马头城)', 117.199, 32.971), ('长淮', 117.463, 32.929),
    ('正阳', 116.527, 32.483), ('瓦埠', 116.924, 32.328), ('武店', 117.290, 32.697), ('炉桥', 117.232, 32.549),
    ('吴圩', 117.466, 32.374), ('殷涧', 117.608, 32.721), ('红心', 117.787, 32.686), ('池河', 117.933, 32.523),
    ('藕塘', 117.925, 32.382), ('章广', 117.913, 32.257), ('管店', 118.071, 32.662), ('女山湖', 118.223, 32.949),
    ('张八岭', 118.213, 32.506), ('建阳 (滁)', 118.314, 32.303), ('来安', 118.427, 32.418), ('汊河', 118.589, 32.225),
    ('全椒', 118.274, 32.073), ('石杨', 118.280, 31.943), ('含山', 118.096, 31.737), ('历阳', 118.356, 31.722),
    ('盱眙', 118.52, 32.94), ('东阳', 118.62, 32.80), ('曲阳', 117.83, 32.92)]
GEM_TOWNS = [('涂山 (禹会诸侯)', 117.232, 32.912), ('八公山', 116.835, 32.635), ('琅琊山', 118.264, 32.284),
             ('老嘉山', 118.234, 32.624), ('江东', 118.60, 31.80)]   # 江东子弟: only the 乌江 ferry reaches it
OIL = []
TECH = [('CAOUTP', '霸王祠 (乌江亭)', 118.465, 31.835, 4, 3),
        ('CAOUTP', '阴陵 (田父绐曰"左")', 117.36, 32.66, 4, 3)]
CITIES = []
BRIDGES = [  # ferries of the time across the 淮水, and the 乌江 ferry across the 长江
    ('下蔡渡', 116.795, 32.708),
    ('荆涂峡', 117.205, 32.950),
    ('钟离渡 (项羽渡淮)', 117.625, 32.915),
    ('虹 (浍口)', 117.895, 33.125),
    ('盱眙', 118.46, 32.975),
    ('乌江渡', 118.50, 31.86),
]
P = {
    '符离': (116.972, 33.754), '蕲': (117.091, 33.516), '灵璧': (117.553, 33.543), '垓下': (117.55, 33.29),
    '谷阳': (117.315, 33.322), '蒙': (116.566, 33.258), '龙亢': (116.890, 33.136), '下蔡': (116.706, 32.712),
    '寿春': (116.778, 32.578), '当涂': (117.199, 32.971), '钟离': (117.62, 32.86), '钟离渡': (117.625, 32.93),
    '阴陵': (117.36, 32.61), '大泽': (117.48, 32.48), '东城': (117.70, 32.30), '历阳': (118.356, 31.722),
    '乌江': (118.40, 31.88), '全椒': (118.274, 32.073), '建阳': (118.314, 32.303), '虹': (117.656, 33.084),
    '夏丘': (117.863, 33.468), '盱眙': (118.52, 32.94), '曲阳': (117.83, 32.92), '池河': (117.933, 32.523),
    '东阳': (118.62, 32.80), '洨': (116.896, 33.425),
}


def _road(*names):
    return [P[n] for n in names]


TRACKS = {  # dirt roads of the time
    '项羽南走': _road('垓下', '钟离渡', '钟离', '阴陵', '大泽', '东城', '历阳', '乌江'),   # the flight of the 800 riders
    '彭城-符离-蕲-垓下': [(116.95, 33.86)] + _road('符离', '蕲', '谷阳', '垓下'),
    '灵璧-垓下': _road('灵璧', '垓下'),
    '彭城-灵璧-夏丘': [(117.40, 33.86)] + _road('灵璧', '夏丘'),
    '蒙-龙亢-当涂': _road('蒙', '龙亢', '当涂', '钟离渡'),
    '蒙-洨-蕲': _road('蒙', '洨', '蕲'),
    '寿春-下蔡': _road('寿春', '下蔡'),
    '寿春-阴陵-东城': _road('寿春', '阴陵'),
    '垓下-虹-盱眙': _road('垓下', '虹', '曲阳', '盱眙', '东阳'),
    '钟离-池河-建阳': _road('钟离', '池河', '建阳', '全椒', '历阳'),
    '东城-全椒': _road('东城', '全椒'),
}
TRACK_COVER = 'SAND'   # in the snow theater: a trodden strip of frozen grass
ROADS = TRACKS
_W = 'data/osm_chuhan6_gaixia/water.json'
RIVER_NAMES = set()
RIVER_BBOX = (116.0, 31.0, 119.0, 34.2)
# no modern canals (怀洪新河, 茨淮新河, 新汴河, 新濉河, 驷马山引江水道 ...)
EXTRA_RIVERS = osmwater.lines(_W, {'涡河', '沱河', '北沱河', '浍河', '濠河', '北淝河', '东淝河', '西淝河', '池河', '滁河',
                                  '窑河', '芡河', '澥河', '天河', '唐河', '清流河', '襄河', '得胜河', '淠河', '濉河', '老濉河',
                                  '小溪河', '姑溪河', '潼河'})
WATER_RIVERS = osmwater.lines(_W, {'淮河'})
WATER_HALF_WIDTH = 0.9
YANGTZE = osmwater.lines(_W, {'长江'})
LAKES = [  # pools of the great marsh at 阴陵 ("陷大泽中")
    (117.43, 32.54, 2.8, 1.9), (117.53, 32.46, 2.0, 2.6), (117.37, 32.45, 2.0, 1.7), (117.59, 32.56, 1.9, 1.5)]
WATER_LEVEL = 0
MIN_WATER_CORNERS, MIN_LAND_CORNERS = 3, 8   # small marsh pools survive
MOUNTAIN_TREES = ['TREE25', 'TREE26', 'TREE27', 'TREE05', 'TREE06', 'TREE07', 'TREE08', 'TREE15']
VALLEY_TREES = ['TREE01', 'TREE02', 'TREE03', 'TREE04', 'TREE10', 'TREE11', 'TREE12', 'TREE17']
CHECK_POINTS = [(n, lon, lat) for n, lon, lat in GEM_TOWNS]
MARSH = (117.46, 32.50, 0.13, 0.09)   # lon, lat, half-widths (deg): the 阴陵 大泽


def water_corners(g):
    """The 长江 is far wider than the 淮水."""
    return g.polyline_distance(YANGTZE, offset=0.0) < 2.2


def ground_cover(g):
    E, R, lon, lat, n1, n2 = g.ELEV_CELL, g.RELIEF_CELL, g.CELL_LON, g.CELL_LAT, g.NOISE, g.NOISE2
    cover = np.full(E.shape, g.GRASS)
    p = np.full(E.shape, 0.008)
    hills = (E > 90) | (R > 18)                                  # 江淮 hills: 涂山, 老嘉山, 张八岭, 琅琊山, 含山
    plain = ~hills
    mlon, mlat, mx, my = MARSH
    marsh = ((lon - mlon) / mx) ** 2 + ((lat - mlat) / my) ** 2 < 1
    riparian = (g.RIVER_D < 1.3) & plain

    cover[plain & (n1 > 0.72)] = g.THICK                         # stubble of the winter fields through the snow
    cover[hills & (n2 > 0.55)] = g.ROUGH                         # snowy rock on the hills
    cover[marsh & (n2 > 0.3)] = g.THICK                          # frozen reed beds of the great marsh
    p[plain] = 0.008
    p[riparian] = 0.07
    p[hills] = 0.10 + 0.22 * (1 - n1[hills])
    p[(E > 200)] = 0.30 + 0.2 * (1 - n2[E > 200])                # wooded tops: 琅琊山, 皇甫山
    p[marsh] = 0.02
    return cover, hills, p


# --- buildings ---------------------------------------------------------------------------
from regions.pools import POOLS as CIVILIAN_POOLS   # noqa: E402

CIVILIAN_START_CLEAR = 7
FLAT_CITIES = []
PROP_DENSITY = 0.2
CIVILIAN_ZONES = [  # lon, lat, radius (cells), pool, density: hamlets of the 淮 plain, no towns
    (117.25, 33.45, 5, 'farm', 0.2),
    (117.85, 33.30, 5, 'farm', 0.2),
    (117.10, 33.10, 5, 'farm', 0.2),
    (117.00, 32.80, 4, 'village', 0.25),
    (117.95, 32.70, 5, 'farm', 0.2),
    (118.40, 32.60, 4, 'village', 0.2),
    (117.30, 32.35, 4, 'village', 0.2),
    (118.10, 32.10, 5, 'farm', 0.2),
    (118.15, 31.85, 4, 'village', 0.25),
    (118.62, 31.88, 3, 'village', 0.3),          # 江东
]
CIVILIAN_FILL = None
