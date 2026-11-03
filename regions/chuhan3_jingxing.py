"""Chu-Han 3: Battle of Jingxing (井陉之战 / 背水一战, October 204 BC). 韩信 came east over the 太行 through the
井陉 defile, where "车不得方轨，骑不得成列"; 陈余 waited at 井陉口 with the 赵 army and would not cut the defile as
李左车 urged. 韩信 sent 2000 light horse with red flags along a hidden path over 抱犊山 to take the empty 赵 camp,
set 10,000 men with their backs to the 绵蔓水, and broke the 赵 army between them.

The 太行 is a plateau of cliffs. One long, narrow road runs down the 绵河 gorge from 阳泉 / the 故关 to the 井陉
basin, with the river on its south side; in the basin the 绵蔓水 (今 冶河) turns north, so the army coming out of
the defile has to ford it (微水, 威州) before the short 井陉口 gorge leads out to the 华北 plain at 土门. A second,
tiny path climbs from the east bank over 抱犊山 to the plain beside the 赵 camp. The 绵蔓水 flows on through its
gorge to the 滹沱河, which crosses the plain past 东垣: one river line from the west edge to the east edge."""
import numpy as np

NAME = 'Chu-Han 3: Battle of Jingxing'
OUTFILE = 'chuhan3_jingxing.mpr'
SEED = 204  # 204 BC
W, H = 180, 152
LX, LY, LW, LH = 2, 4, 176, 139
LON0, LON1, LAT0, LAT1 = 113.50, 114.65, 38.37, 37.6492
DEM_DIR, DEM_ZOOM = 'data/dem_chuhan3_jingxing', 11
BLUR_SMOOTH, BLUR_WIDE, BLUR_TERRAIN = 1, 5, 2
ELEV_PTS = [0, 120, 260, 650, 900]          # the plain and the river floors at level 0; the 阳泉 uplands a step up
LVL_PTS = [0, 0, 1, 1, 2]
CLIFF_FOOT = 0                              # every cliff stands on level 0, as the rivers do: water can meet the cliffs
PLATEAU_ELEV_PTS = [250, 500, 800, 1100, 1500, 1900]
PLATEAU_LVL_PTS = [4, 6, 8, 10, 12, 13]
PLATEAU_TREES = 0.0
THEATER = 'TEMPERATE'
LIGHTING = dict(Red=1.04, Green=1.0, Blue=0.92, Level=0.03, Ambient=1.02)   # low October sun

# --- the river line: 桃河 -> 绵河 (the defile) -> 绵蔓水 / 冶河 -> 滹沱河, west edge to east edge (OSM, simplified) ---
TAOHE = [(113.48, 37.87), (113.518, 37.876), (113.565, 37.862), (113.61, 37.86), (113.631, 37.846), (113.664, 37.852),
         (113.688, 37.848), (113.7, 37.865), (113.699, 37.884), (113.724, 37.882), (113.742, 37.873), (113.771, 37.872),
         (113.793, 37.887), (113.81, 37.885), (113.799, 37.893), (113.815, 37.902), (113.822, 37.915), (113.839, 37.908),
         (113.835, 37.922), (113.853, 37.927), (113.852, 37.95), (113.864, 37.961)]
MIANHE = [(113.864, 37.961), (113.879, 37.971), (113.889, 37.966), (113.898, 37.974), (113.914, 37.973),
          (113.921, 37.984), (113.945, 37.994), (113.952, 38.003), (113.967, 37.995), (113.98, 37.996),
          (114.001, 38.008), (114.019, 37.994), (114.026, 38.012), (114.045, 38.016), (114.081, 37.999),
          (114.089, 38.007), (114.102, 38.002)]
YEHE = [(114.102, 38.002), (114.123, 38.025), (114.135, 38.063), (114.147, 38.075), (114.135, 38.097),
        (114.133, 38.132), (114.126, 38.136), (114.135, 38.147), (114.119, 38.174), (114.138, 38.182),
        (114.142, 38.193), (114.156, 38.2), (114.154, 38.208), (114.139, 38.22), (114.142, 38.234), (114.186, 38.255),
        (114.208, 38.29), (114.254, 38.283)]
HUTUO = [(114.254, 38.283), (114.278, 38.254), (114.305, 38.25), (114.317, 38.264), (114.333, 38.27), (114.38, 38.27),
         (114.403, 38.25), (114.412, 38.211), (114.54, 38.124), (114.591, 38.109), (114.626, 38.109), (114.675, 38.098)]
WATER_RIVERS = [TAOHE + MIANHE[1:] + YEHE[1:] + HUTUO[1:]]
WATER_HALF_WIDTH = 0.9
LAKES = []
WATER_LEVEL = 0
MIN_WATER_CORNERS, MIN_LAND_CORNERS = 6, 8
BRIDGES = [  # fords and the later crossings
    ('阳泉', 113.585, 37.861),
    ('微水渡 (背水阵)', 114.127, 38.035),
    ('威州渡', 114.137, 38.105),
    ('平山', 114.30, 38.252),
    ('东垣 (真定)', 114.545, 38.122),
]

# --- the passes ------------------------------------------------------------------------------------------
# The defile road keeps to the north bank of the 绵河 (the river hugs the southern cliffs): at each step it runs
# a fixed distance north of the highest point of the river's meanders nearby.
def _north_of(line, lon0, lon1, step=0.008, reach=0.009, gap=0.0085):
    pts = []
    for a, b in zip(line, line[1:]):
        pts += [(a[0] + (b[0] - a[0]) * k / 20, a[1] + (b[1] - a[1]) * k / 20) for k in range(20)]
    out = []
    for lo in np.arange(lon0, lon1 + 1e-9, step):
        near = [la for lo_, la in pts if abs(lo_ - lo) <= reach]
        out.append((round(float(lo), 4), round(max(near) + gap, 4)))
    return out


DEFILE = [(113.84, 37.975)] + _north_of(MIANHE, 113.872, 114.096) + [(114.103, 38.032)]
PASSES = {
    '井陉道': DEFILE,                                                          # 故关 -> 天长 -> 北横口, ~25 km
    '井陉口': [(114.15, 38.035), (114.175, 38.05), (114.202, 38.063), (114.225, 38.055), (114.249, 38.048),
            (114.268, 38.066), (114.29, 38.072)],                                 # 微水 -> 东天门 -> 上安 -> 土门
    '抱犊间道': [(114.152, 38.098), (114.172, 38.11), (114.196, 38.118), (114.225, 38.126), (114.248, 38.118),
              (114.268, 38.108), (114.288, 38.101), (114.31, 38.102)],          # 韩信's 2000 light horse
}
BASIN = [(113.995, 38.02), (114.0, 38.10), (114.06, 38.122), (114.125, 38.122), (114.145, 38.112), (114.165, 38.095),
         (114.19, 38.075), (114.215, 38.062), (114.205, 38.02), (114.175, 37.99), (114.11, 37.972), (114.06, 37.995),
         (114.03, 38.008)]                       # the 井陉 basin: 矿区 on the west bank, 微水 on the east bank
PASS_WIDTH = {'井陉道': 2.8, '井陉口': 3.0, '抱犊间道': 1.6}             # half widths in cells
PASS_HALF_WIDTH = 3.0                                                          # for connectivity.py BLOCK tests
RIDGES = {}
RIDGE_HALF_WIDTH = 3.0

STARTS = [
    ('阳泉 (韩信东进)', 113.585, 37.905),        # the 汉 army coming from 太原 over the 太行
    ('平定 (上艾)', 113.623, 37.80),              # the road up from 上党, south of the 桃河
    ('韩信营 (井陉)', 114.05, 38.078),           # "未至井陉口三十里，止舍"
    ('背水阵 (绵蔓水东岸)', 114.177, 38.042),    # 10,000 men with their backs to the river
    ('井陉口 赵营 (陈余)', 114.325, 38.05),      # 陈余 / 赵王歇 at the mouth of the defile
    ('东垣', 114.555, 38.062),                    # the 赵 city on the 滹沱河 (later 真定)
    ('泜水 (元氏)', 114.52, 37.745),             # 陈余 was killed on the 泜水
    ('灵寿', 114.37, 38.318),                     # north of the 滹沱河
]
START_FLAT, START_ORE_DIST, START_WATER_CLEAR = 9, 7, 3
MAP_CENTER = (114.15, 38.02)
ORE_TOWNS = [('荫营', 113.565, 37.95), ('杨家庄', 113.647, 37.894),
             ('巨城', 113.72, 37.942), ('岔口', 113.713, 38.029), ('西南舁', 113.62, 38.028), ('冶西', 113.56, 37.765),
             ('锁簧', 113.653, 37.737), ('柏井', 113.80, 37.83), ('娘子关', 113.86, 37.975),
             ('秀林', 114.092, 37.99), ('威州', 114.16, 38.083), ('贾庄', 114.04, 38.11), ('上安', 114.21, 38.07),
             ('获鹿', 114.33, 38.095), ('上庄', 114.385, 38.03), ('铜冶', 114.379, 37.968), ('山尹村', 114.406, 37.918),
             ('姬村', 114.398, 37.87), ('南佐', 114.359, 37.838), ('殷村', 114.466, 37.846), ('苏村', 114.342, 37.792),
             ('张楞', 114.329, 37.745), ('赵同', 114.497, 37.781), ('马村', 114.509, 37.858), ('窦妪', 114.505, 37.899),
             ('方村', 114.553, 37.979), ('宋营', 114.62, 38.013), ('平山', 114.195, 38.262), ('李村', 114.341, 38.196),
             ('宜安', 114.29, 38.18), ('南岗', 114.501, 38.19), ('三圣院', 114.43, 38.29), ('化皮', 114.559, 38.31)]
GEM_TOWNS = [('故关', 113.835, 37.97),               # the old 井陉 gate, the defile's west end
             ('抱犊寨', 114.205, 38.122),            # on the hidden path
             ('封龙山', 114.36, 37.935),             # 李左车 / the 赵 hills south of the defile mouth
             ('泜水', 114.47, 37.72)]                # where 陈余 died
OIL = []
TECH = [('CAOUTP', '赵壁 (拔赵帜立汉赤帜)', 114.318, 38.115, 4, 3),   # the 赵 camp the light horse took
        ('CAOUTP', '东天门 (秦驰道)', 114.18, 38.075, 4, 3)]          # the old post road over the 井陉口 ridge
LANDMARKS = []
CITIES = []
TRACKS = {  # the 秦 post road through the defile, and the road along the foot of the 太行
    '井陉古道': [(113.585, 37.905), (113.70, 37.935), (113.80, 37.965)] + DEFILE + [(114.115, 38.033), (114.15, 38.035),
             (114.175, 38.05), (114.202, 38.063), (114.225, 38.055), (114.249, 38.048), (114.268, 38.066), (114.29, 38.072),
             (114.325, 38.05), (114.42, 38.055), (114.555, 38.062)],
    '太行东麓道': [(114.37, 38.318), (114.33, 38.27), (114.305, 38.25), (114.30, 38.2), (114.31, 38.12),
              (114.325, 38.05), (114.37, 37.95), (114.43, 37.85), (114.52, 37.745)],
}
TRACK_COVER = 'SAND'
ROADS = dict(PASSES, **TRACKS)
RIVER_NAMES = set()
RIVER_BBOX = (113.0, 37.0, 115.0, 39.0)
from regions import osmwater   # noqa: E402
EXTRA_RIVERS = osmwater.lines('data/osm_chuhan3_jingxing/rivers.json',
                              {'温河', '甘陶河', '松溪河', '南川河', '槐河', '磁河', '太平河', '小作河', '割髭河', '南甸河'})
EXTRA_RIVERS += [[(113.80, 38.33), (113.913, 38.307), (113.972, 38.338), (114.014, 38.313), (114.068, 38.329),
                  (114.122, 38.325), (114.161, 38.299), (114.188, 38.291), (114.21, 38.305), (114.232, 38.306),
                  (114.254, 38.283)]]          # the upper 滹沱河 above the 冶河 mouth
MOUNTAIN_TREES = ['TREE25', 'TREE26', 'TREE27', 'TREE05', 'TREE06', 'TREE07', 'TREE08', 'TREE15']
VALLEY_TREES = ['TREE01', 'TREE02', 'TREE03', 'TREE04', 'TREE10', 'TREE11', 'TREE12', 'TREE17']


def plateau(g):
    """(mountain mask, cells that must stay low, cells that must stay mountain). Low land is the 阳泉 uplands in
    the west, the 井陉 basin and the plain east of the 太行; everything between is mountain, cut only by the
    passes and the river."""
    def grow(m, k=1):
        for _ in range(k):
            m = m | g.shift(m, 1, 0, False) | g.shift(m, -1, 0, False) | g.shift(m, 0, 1, False) | g.shift(m, 0, -1, False)
        return m

    E, lon, lat = g.ELEV_CELL, g.CELL_LON, g.CELL_LAT
    keep = np.zeros_like(g.VALID)
    for name, line in PASSES.items():
        keep |= g.polyline_distance([line]) < PASS_WIDTH[name]
    passes = keep.copy()
    sites = [(lon_, lat_, START_ORE_DIST + 4) for _, lon_, lat_ in STARTS]   # the base and its two ore fields
    sites += [(lon_, lat_, 4) for _, lon_, lat_ in ORE_TOWNS] + [(lon_, lat_, 3.5) for _, lon_, lat_ in GEM_TOWNS]
    sites += [(lon_, lat_, max(w, h) + 2) for _, _, lon_, lat_, w, h in TECH]
    sites += [(lon_, lat_, 7) for _, lon_, lat_ in BRIDGES]                      # both banks of every ford
    for lon_, lat_, r in sites:
        keep |= g.dist_to_point(lon_, lat_) < r
    # soft limits: the threshold falls off past each zone's edge, so the cliffs follow the contours, not a line
    west = E < 900 - 5000 * np.maximum(0, lon - 113.80)
    from matplotlib.path import Path
    inside = Path(BASIN).contains_points(np.column_stack([lon.ravel(), lat.ravel()])).reshape(lon.shape)
    out = np.zeros(lon.shape)                         # cells outside the basin outline, by distance
    m = inside.copy()
    for k in range(1, 7):
        nm = grow(m)
        out[nm & ~m] = k
        m = nm
    out[~m] = 99
    basin = E < 340 - 40 * out
    behind = (lat > 38.075) & (lat < 38.2)                          # the hills the hidden path crosses
    plain = E < 215 - 2500 * np.maximum(0, 114.12 - lon) - np.where(behind, 2500 * np.maximum(0, 114.30 - lon), 0)
    gorge = (lon > 114.09) & (lon < 114.25) & (lat > 38.125) & (lat < 38.238)   # the 绵蔓水 gorge: river only
    low = (west | basin | plain) & ~gorge
    mount = g.VALID & ~low
    mount = ~grow(~grow(mount, 2), 2)                 # close: no slots narrower than ~4 cells
    mount = grow(~grow(~mount, 2), 2)                 # open: no ridges narrower than ~4 cells
    from cliffs import majority
    keep_all = keep | g.water_cell
    mount = majority(majority(mount & ~keep_all, 2), 3) & ~keep_all
    low = g.VALID & ~mount & ~g.water_cell        # land only: the rivers split the banks
    lab, sizes = g.components(low)
    bases = np.zeros_like(low)
    for _, lon_, lat_ in STARTS:
        bases |= g.dist_to_point(lon_, lat_) < 3
    reach = np.unique(lab[(passes | bases) & low])
    mount |= low & ~np.isin(lab, reach)               # land no army can walk to from a base joins the mountain
    lab, sizes = g.components(mount)
    mount &= (lab >= 0) & (sizes[np.maximum(lab, 0)] >= 60)
    return mount, keep_all, np.zeros_like(mount)


def ground_cover(g):
    """October 204 BC: millet stubble on the plain, bare loess and rock in the 太行, oak and pine on the heights."""
    E, R, lon, lat, n1, n2 = g.ELEV_CELL, g.RELIEF_CELL, g.CELL_LON, g.CELL_LAT, g.NOISE, g.NOISE2
    cover = np.full(E.shape, g.GRASS)
    plain = E < 300
    hills = (E >= 300) & (E < 900)
    high = E >= 900
    riparian = g.RIVER_D < 1.3
    cover[plain & (n1 > 0.45)] = g.THICK                       # harvested millet and sorghum
    cover[plain & (n1 <= 0.45) & (n2 > 0.75)] = g.SAND         # threshing floors, fallow
    cover[hills & (n1 > 0.4)] = g.SAND                          # loess terraces
    cover[hills & (n1 <= 0.4) & (n2 > 0.45)] = g.ROUGH          # limestone scree
    cover[high & (n2 > 0.4)] = g.ROUGH
    p = np.full(E.shape, 0.008)
    p[plain] = 0.01
    p[riparian] = 0.1                                           # willows along the rivers
    p[hills] = 0.03 + 0.05 * n2[hills]
    p[high] = 0.12 + 0.3 * n1[high]                             # oak and pine on the 太行
    road = g.polyline_distance(PASSES.values())
    p[road < 4] *= 0.3
    p[road < 2.2] = 0
    mountain = hills | high
    return cover, mountain, p


# --- buildings ---------------------------------------------------------------------------
from regions.pools import POOLS as CIVILIAN_POOLS   # noqa: E402

CIVILIAN_START_CLEAR = 7
FLAT_CITIES = []
PROP_DENSITY = 0.2
CIVILIAN_ZONES = [  # lon, lat, radius (cells), pool, density: farming villages only
    (113.62, 37.82, 3, 'village', 0.4),          # 平定 (then 上艾)
    (114.10, 38.05, 3, 'farm', 0.25),            # the 井陉 basin
    (114.42, 37.98, 5, 'farm', 0.25),            # the plain below the 太行
    (114.45, 37.82, 5, 'farm', 0.25),
    (114.50, 38.20, 4, 'farm', 0.25),
    (114.60, 38.00, 4, 'village', 0.3),
]
CIVILIAN_FILL = None
