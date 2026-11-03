"""楚汉争霸 1 - 还定三秦 (Return to Guanzhong, Aug 206 BC): "明修栈道，暗度陈仓". 刘邦, made King of Han and sent to
the 汉中 basin, burns the 褒斜 plank road behind him; four months later 韩信 sends 樊哙 to rebuild it in plain sight
while the army slips over the 故道 by 散关 and falls on 章邯 at 陈仓. The 秦岭 between the 汉中 basin and the 关中
plain of the Wei is a plateau ringed by cliffs that ground units cannot climb; the only ways across are the four
ancient roads through it, each a canyon traced along its real valleys: 故道 (陈仓 - 散关 - 凤县 - 嘉陵江 - 略阳 -
沔阳, the road 韩信 took), 褒斜道 (斜谷 - 太白 - 褒谷 - 石门, the burned 栈道, the narrowest), 傥骆道 (骆谷 - 华阳 -
傥水 - 洋县) and 子午道 (子午谷 - 宁陕 - 子午河 - 汉水). In 关中 wait the three Qin kings: 章邯 at 废丘, 司马欣 at
栎阳, and the burned ruins of 咸阳."""
import numpy as np

NAME = 'Chu-Han 1: Return to Guanzhong'
OUTFILE = 'chuhan1_sanqin.mpr'
SEED = 206  # 206 BC
W, H = 200, 136
LX, LY, LW, LH = 2, 4, 196, 123
LON0, LON1, LAT0, LAT1 = 105.92, 109.40, 34.76, 32.9316
DEM_DIR, DEM_ZOOM = 'data/dem_chuhan1_sanqin', 9
BLUR_SMOOTH, BLUR_WIDE, BLUR_TERRAIN = 1, 5, 2
THEATER = 'TEMPERATE'   # August: harvest time in 关中, rice in 汉中; the cliffs need the temperate set
ELEV_PTS = [300, 450, 620, 800]                   # Wei plain and 汉中 basin at level 0-1, the 周原 / 雍 terraces at 2
LVL_PTS = [0, 0, 1, 2]
CLIFF_FOOT = 2                                    # every cliff stands on level 2; its top is level 6
PLATEAU_ELEV_N, PLATEAU_ELEV_S = 860, 760         # the mountain contour north and south of the 秦岭 crest (m)
PLATEAU_ELEV_PTS = [760, 1300, 1900, 2600, 3300]
PLATEAU_LVL_PTS = [6, 8, 10, 12, 13]              # 太白山 (3771 m) at level 13
PLATEAU_TREES = 0.0
LIGHTING = dict(Red=1.04, Green=1.0, Blue=0.93, Level=0.03, Ambient=1.04)   # late summer

PASSES = {  # the four roads over the 秦岭, traced along their valleys on the DEM (least-cost paths)
    '故道': [(107.249, 34.362), (107.133, 34.357), (107.029, 34.271), (106.985, 34.271), (106.902, 34.239),
           (106.859, 34.203), (106.815, 34.198), (106.787, 34.18), (106.76, 34.157), (106.76, 34.144),
           (106.738, 34.121), (106.738, 34.103), (106.661, 34.03), (106.661, 33.98), (106.518, 33.911),
           (106.474, 33.907), (106.447, 33.88), (106.447, 33.866), (106.419, 33.852), (106.419, 33.838),
           (106.397, 33.82), (106.375, 33.82), (106.353, 33.802), (106.326, 33.797), (106.249, 33.738),
           (106.21, 33.738), (106.194, 33.724), (106.194, 33.688), (106.21, 33.674), (106.205, 33.656),
           (106.111, 33.605), (106.078, 33.578), (106.062, 33.578), (106.035, 33.541), (106.046, 33.413),
           (106.062, 33.367), (106.084, 33.344), (106.238, 33.312), (106.293, 33.312), (106.304, 33.321),
           (106.348, 33.317), (106.419, 33.271), (106.43, 33.234), (106.474, 33.193), (106.6, 33.165),
           (106.611, 33.156), (106.672, 33.156)],
    '褒斜道': [(107.672, 34.194), (107.644, 34.166), (107.65, 34.107), (107.633, 34.098), (107.633, 34.085),
            (107.611, 34.062), (107.452, 34.062), (107.441, 34.053), (107.298, 34.057), (107.216, 34.021),
            (107.216, 33.957), (107.199, 33.939), (107.15, 33.939), (107.128, 33.925), (107.117, 33.884),
            (107.095, 33.852), (107.095, 33.825), (107.062, 33.793), (107.056, 33.729), (107.018, 33.701),
            (106.996, 33.651), (106.996, 33.578), (106.979, 33.56), (106.979, 33.532), (106.996, 33.505),
            (106.99, 33.395), (106.974, 33.344), (106.93, 33.303), (106.93, 33.275), (106.952, 33.257),
            (106.957, 33.207), (106.924, 33.179)],
    '傥骆道': [(108.105, 34.089), (108.144, 34.089), (108.188, 34.053), (108.188, 34.039), (108.177, 34.003),
            (108.149, 33.971), (108.144, 33.943), (108.073, 33.87), (108.051, 33.87), (108.018, 33.893),
            (107.974, 33.898), (107.941, 33.889), (107.902, 33.857), (107.776, 33.843), (107.721, 33.816),
            (107.534, 33.825), (107.452, 33.743), (107.452, 33.733), (107.507, 33.692), (107.507, 33.669),
            (107.529, 33.624), (107.529, 33.573), (107.518, 33.56), (107.43, 33.514), (107.452, 33.431),
            (107.474, 33.409), (107.474, 33.395), (107.518, 33.363), (107.529, 33.22)],
    '子午道': [(108.891, 34.048), (108.809, 34.053), (108.754, 34.007), (108.765, 33.989), (108.765, 33.939),
            (108.737, 33.907), (108.737, 33.893), (108.693, 33.848), (108.693, 33.82), (108.71, 33.802),
            (108.699, 33.729), (108.638, 33.665), (108.6, 33.674), (108.534, 33.729), (108.419, 33.724),
            (108.397, 33.743), (108.364, 33.743), (108.325, 33.743), (108.314, 33.733), (108.309, 33.679),
            (108.287, 33.651), (108.27, 33.551), (108.254, 33.537), (108.248, 33.473), (108.232, 33.459),
            (108.232, 33.418), (108.149, 33.353), (108.105, 33.344), (108.073, 33.317), (108.067, 33.239),
            (108.056, 33.216), (108.023, 33.188), (107.99, 33.184), (107.963, 33.161), (107.946, 33.174),
            (107.891, 33.174), (107.836, 33.197), (107.748, 33.188), (107.567, 33.207), (107.54, 33.229)],
}
PASS_HALF_WIDTH = 3.5
PASS_WIDTH = {'褒斜道': 3.0}                       # the burned plank road: a narrow canyon
PASS_RIM = 1.5                                    # the outer cells of a canyon may take cliff pieces
RIDGES = {}
RIDGE_HALF_WIDTH = 3.0

STARTS = [
    ('南郑', 107.02, 33.07),           # 汉王刘邦's capital, 拜将坛
    ('城固', 107.33, 33.16),
    ('沔阳', 106.67, 33.155),          # 勉县: the 故道 comes down the 嘉陵江 to here
    ('陈仓', 107.25, 34.36),           # 暗度陈仓: 韩信 beats 章邯 here
    ('郿', 107.75, 34.28),             # mouth of the 斜谷, north end of the 褒斜道
    ('废丘', 108.49, 34.30),           # 雍王章邯 (槐里, east of 兴平)
    ('咸阳', 108.858, 34.414),         # the Qin capital, burned by 项羽 a few months before
    ('栎阳', 109.208, 34.585),         # 塞王司马欣
]
START_FLAT, START_ORE_DIST, START_WATER_CLEAR = 9, 7.5, 3
MAP_CENTER = (107.8, 33.8)
ORE_TOWNS = [  # 关中
             ('雍', 107.39, 34.52), ('虢', 107.50, 34.37), ('美阳', 108.21, 34.26), ('盩厔', 108.20, 34.16),
             ('鄠', 108.60, 34.11), ('杜', 108.95, 34.20), ('霸上', 109.05, 34.27), ('泾阳', 108.83, 34.53),
             ('池阳', 108.93, 34.65), ('斄', 108.00, 34.30), ('高陵', 109.05, 34.50), ('阿房', 108.70, 34.22),
             # 汉中
             ('褒中', 106.98, 33.20), ('成固东', 107.45, 33.20), ('洋县', 107.62, 33.24), ('南郑南', 106.92, 33.01),
             ('沔阳东', 106.82, 33.12), ('黄金峡', 107.92, 33.15),
             # along the roads
             ('凤县', 106.518, 33.911), ('两当', 106.375, 33.82), ('略阳', 106.084, 33.344), ('留坝', 106.996, 33.578),
             ('厚畛子', 107.90, 33.86), ('江口', 108.70, 33.78), ('子午镇', 108.07, 33.30)]
GEM_TOWNS = [('散关', 107.02, 34.272),            # 大散关, 韩信's road over the crest
             ('太白', 107.30, 34.057),            # the top of the 褒斜道
             ('华阳', 107.53, 33.60),             # 傥骆道
             ('宁陕', 108.27, 33.55)]             # 子午道
OIL = []
TECH = [('CAOUTP', '斜谷关', 107.668, 34.18, 4, 3),     # north gate of the 褒斜道
        ('CAOUTP', '子午关', 108.885, 34.035, 4, 3),    # north gate of the 子午道
        ('CAOUTP', '石门', 106.93, 33.21, 4, 3)]       # 石门: the south mouth of the 褒斜 plank road
LANDMARKS = []
CITIES = [('咸阳', 108.858, 34.414, 2.2), ('南郑', 107.02, 33.07, 1.8), ('废丘', 108.49, 34.30, 1.4),
          ('栎阳', 109.208, 34.585, 1.4), ('陈仓', 107.25, 34.36, 1.2), ('雍', 107.39, 34.52, 1.2),
          ('城固', 107.33, 33.16, 1), ('沔阳', 106.67, 33.155, 1), ('郿', 107.75, 34.28, 1)]
CITY_COVER = 493                                  # rammed earth (temperate SAND), not pavement
ROADS_PLAIN = {  # 秦 roads on the plains
    '渭北道': [(107.249, 34.362), (107.50, 34.37), (107.75, 34.28), (108.00, 34.30), (108.21, 34.26),
            (108.49, 34.30), (108.858, 34.414), (109.05, 34.50), (109.208, 34.585), (109.40, 34.62)],
    '雍道': [(107.249, 34.362), (107.39, 34.52)],
    '汉中道': [(106.672, 33.156), (106.82, 33.12), (107.02, 33.07), (107.33, 33.16), (107.54, 33.229)],
    '褒中道': [(106.924, 33.179), (107.02, 33.07)],
}
TRACKS = dict(PASSES, **ROADS_PLAIN)              # drawn as dirt roads
TRACK_COVER = 'SAND'
ROADS = TRACKS                                    # no trees on them
RIVER_NAMES = set()
RIVER_BBOX = (105.5, 32.5, 110.0, 35.0)
from regions import osmwater   # noqa: E402
EXTRA_RIVERS = osmwater.lines('data/osm_chuhan1_sanqin/rivers.json',
                              {'渭河', '汉江', '褒河', '嘉陵江', '石头河', '酉水河', '傥水河', '子午河', '千河', '泾河',
                               '沣河', '湑水河', '清姜河', '红岩河', '太白河'})
WATER_RIVERS, WATER_HALF_WIDTH, LAKES = [], 0, []
WATER_LEVEL = 0
MOUNTAIN_TREES = ['TREE25', 'TREE26', 'TREE27', 'TREE05', 'TREE06', 'TREE07', 'TREE08', 'TREE15']
VALLEY_TREES = ['TREE01', 'TREE02', 'TREE03', 'TREE04', 'TREE10', 'TREE11', 'TREE12', 'TREE17']
CHECK_POINTS = [('散关', 107.02, 34.272), ('太白', 107.30, 34.057), ('华阳', 107.53, 33.60), ('宁陕', 108.27, 33.55),
                ('凤县', 106.518, 33.911), ('留坝', 106.996, 33.578), ('略阳', 106.084, 33.344)]


def plateau(g):
    """(mountain mask, cells that must stay low, cells that must stay mountain). Low land is only the 关中 plain and
    the 汉中 basin (the low ground around the starts, below the contour and smoothed); every valley in the range
    is mountain except the four road canyons and the towns, fields and building sites along them."""
    def grow(m, k=1):
        for _ in range(k):
            m = m | g.shift(m, 1, 0, False) | g.shift(m, -1, 0, False) | g.shift(m, 0, 1, False) | g.shift(m, 0, -1, False)
        return m

    passes = np.zeros_like(g.VALID)                   # the canyon: low ground ...
    core = np.zeros_like(g.VALID)                     # ... whose middle must stay low; the cliffs may take the rim
    for name, line in PASSES.items():
        d = g.polyline_distance([line])
        passes |= d < PASS_WIDTH.get(name, PASS_HALF_WIDTH)
        core |= d < PASS_WIDTH.get(name, PASS_HALF_WIDTH) - PASS_RIM
    keep = core.copy()
    sites = [(lon, lat, 8) for _, lon, lat in STARTS]
    sites += [(lon, lat, 4) for _, lon, lat in ORE_TOWNS] + [(lon, lat, 3.5) for _, lon, lat in GEM_TOWNS]
    sites += [(lon, lat, max(w, h) + 2) for _, _, lon, lat, w, h in TECH + LANDMARKS]
    for lon, lat, r in sites:
        keep |= g.dist_to_point(lon, lat) < r
    north = g.CELL_LAT > 33.75                        # the 秦岭 crest runs about along 33.75-34.0 N
    mount = (g.ELEV_CELL > np.where(north, PLATEAU_ELEV_N, PLATEAU_ELEV_S)) & g.VALID
    mount = ~grow(~grow(mount, 2), 2)                 # close: valleys narrower than ~4 cells fill up
    mount = grow(~grow(~mount, 2), 2)                 # open: no ridges narrower than ~4 cells
    low = g.VALID & ~mount                            # only the basins around the starts stay low
    lab, sizes = g.components(low)
    basins = np.unique([lab[int(x), int(y)] for x, y in (g.to_iso(lon, lat) for _, lon, lat in STARTS)])
    mount = g.VALID & ~np.isin(lab, basins[basins >= 0])
    mount &= ~keep & ~passes
    from cliffs import majority
    mount = majority(majority(mount, 2), 3) & ~keep & ~passes   # a smooth outline the cliff pieces can follow
    low = g.VALID & ~mount                            # pockets cut off by the smoothing go back to the mountain
    lab, sizes = g.components(low)
    reach = np.unique([lab[int(x), int(y)] for x, y in (g.to_iso(lon, lat) for _, lon, lat in STARTS)])
    mount |= low & ~np.isin(lab, reach)
    lab, sizes = g.components(mount)
    mount &= (lab >= 0) & (sizes[np.maximum(lab, 0)] >= 60)
    return mount, keep, np.zeros_like(keep)


def ground_cover(g):
    """August 206 BC: stubble and threshing floors on the 关中 loess, green rice in 汉中, forest on the 秦岭."""
    E, R, lon, lat, n1, n2 = g.ELEV_CELL, g.RELIEF_CELL, g.CELL_LON, g.CELL_LAT, g.NOISE, g.NOISE2
    cover = np.full(E.shape, g.GRASS)
    guanzhong = (lat > 33.9) & (E < 900)
    hanzhong = (lat <= 33.9) & (E < 760)
    mountain = ~guanzhong & ~hanzhong
    alpine = E > 2800
    riparian = g.RIVER_D < 1.3
    cover[guanzhong & (n1 > 0.5)] = g.SAND                    # harvested millet and wheat, yellow loess
    cover[guanzhong & (n1 <= 0.5) & (n2 > 0.6)] = g.THICK
    cover[hanzhong & (n1 > 0.4)] = g.THICK                    # paddies
    cover[mountain & (n2 > 0.65)] = g.ROUGH
    cover[alpine & (n1 > 0.3)] = g.ROUGH                      # 太白山 above the tree line
    p = np.full(E.shape, 0.01)
    p[guanzhong] = 0.008
    p[hanzhong] = 0.015
    p[riparian] = 0.1
    p[mountain] = 0.2 + 0.3 * n1[mountain]                    # the 秦岭 forest
    p[alpine] = 0.03
    road = g.polyline_distance(PASSES.values())
    p[road < PASS_HALF_WIDTH + 1] *= 0.3                       # keep the canyons open for armies
    p[road < 2.5] = 0
    return cover, mountain, p


# --- buildings ---------------------------------------------------------------------------
from regions.pools import POOLS as CIVILIAN_POOLS   # noqa: E402

CIVILIAN_START_CLEAR = 7
FLAT_CITIES = []
PROP_DENSITY = 0.2
CIVILIAN_ZONES = [  # lon, lat, radius (cells), pool, density
    (107.90, 34.32, 5, 'farm', 0.25),            # 周原
    (108.30, 34.30, 5, 'farm', 0.25),
    (108.75, 34.25, 5, 'farm', 0.25),
    (108.95, 34.55, 5, 'farm', 0.25),
    (107.45, 34.42, 4, 'village', 0.3),
    (106.85, 33.12, 4, 'farm', 0.3),             # 汉中 paddies
    (107.18, 33.12, 4, 'farm', 0.3),
    (107.48, 33.20, 3, 'village', 0.3),
]
CIVILIAN_FILL = ('village', 0.5)
