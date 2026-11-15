"""Defense of the Great Wall (长城抗战, 1 Jan - 31 May 1933). After 热河 fell in ten days and 承德 was given up on
4 March, the Kwantung Army came down from the 燕山 to the passes of the Ming wall: 山海关 (taken 3 Jan), 古北口 and
南天门 (中央军 第17军, March - May), 喜峰口 and 罗文峪 (29军, the broadsword night raids of March), 冷口 and 界岭口
(32军). In May the Japanese crossed the 滦河 and the war ended with the 塘沽 truce.
The 燕山 is a plateau ringed by cliffs that ground units cannot climb, and the Great Wall (OpenStreetMap,
data/osm_changcheng/wall.json) runs along it as a ridge that stays mountain everywhere except at the gates; the only
ground routes between 热河 and the 冀东 plain are the pass roads, traced as least-cost valley paths on the DEM.
Sandbag works stand on the wall line across each gate, with the gate itself left open. The lower 滦河 is real water
with three crossings; the 北宁 railway runs along the coast to 山海关."""
import json
import os

import numpy as np

from regions import osmwater

NAME = 'Great Wall 1933'
OUTFILE = 'changcheng.mpr'
SEED = 1933
W, H = 220, 140
LX, LY, LW, LH = 2, 4, 216, 127
LON0, LON1, LAT0, LAT1 = 116.65, 119.98, 41.12, 39.62
DEM_DIR, DEM_ZOOM = 'data/dem_changcheng', 10
BLUR_SMOOTH, BLUR_WIDE, BLUR_TERRAIN = 1, 5, 2
THEATER = 'TEMPERATE'                       # March 1933: bare fields and the last snow; the cliffs need the temperate set
ELEV_PTS = [0, 120, 300]                    # the 冀东 plain and the valley floors at level 0, the foothills a step up
LVL_PTS = [0, 0, 1]
CLIFF_FOOT = 0                              # every cliff stands on level 0, as the 滦河 does: water can meet the cliffs
PLATEAU_ELEV = 220                          # the mountain mask follows this contour (m)
PLATEAU_ELEV_PTS = [220, 500, 800, 1200, 1600, 2000]
PLATEAU_LVL_PTS = [4, 6, 8, 10, 12, 13]     # 雾灵山 (2118 m) at level 13
PLATEAU_TREES = 0.0
LIGHTING = dict(Red=0.97, Green=0.97, Blue=1.04, Level=0.02, Ambient=0.95)   # cold early-spring light

_LUOWENYU_NORTH = [  # 半壁山 to 承德县 (下板城), where it meets the road from 承德 down to 宽城
    (117.949, 40.369), (117.974, 40.369), (117.999, 40.378), (118.023, 40.384), (118.048, 40.39), (118.073, 40.403),
    (118.077, 40.422), (118.081, 40.441), (118.093, 40.459), (118.11, 40.478), (118.089, 40.494), (118.073, 40.51),
    (118.06, 40.528), (118.04, 40.541), (118.015, 40.547), (118.007, 40.553), (118.015, 40.572), (118.019, 40.591),
    (118.032, 40.61), (118.056, 40.619), (118.081, 40.619), (118.106, 40.625), (118.13, 40.632), (118.155, 40.638),
    (118.18, 40.65), (118.18, 40.669), (118.163, 40.682), (118.163, 40.7), (118.143, 40.707), (118.135, 40.722),
    (118.135, 40.741), (118.147, 40.76), (118.151, 40.769)]
PASSES = {  # the roads through the 燕山, traced along their valleys on the DEM (least-cost paths)
    '古北口道': [(116.837, 40.375), (116.862, 40.375), (116.886, 40.381), (116.911, 40.397), (116.936, 40.412),
             (116.94, 40.431), (116.956, 40.45), (116.96, 40.469), (116.985, 40.488), (117.01, 40.506), (117.035, 40.525),
             (117.059, 40.544), (117.084, 40.563), (117.092, 40.582), (117.113, 40.597), (117.125, 40.616),
             (117.138, 40.635), (117.15, 40.654), (117.15, 40.672), (117.158, 40.691), (117.162, 40.71),
             (117.187, 40.725), (117.187, 40.741), (117.191, 40.76), (117.199, 40.778), (117.199, 40.797),
             (117.216, 40.816), (117.224, 40.835), (117.245, 40.853), (117.269, 40.869), (117.29, 40.888),
             (117.311, 40.906), (117.311, 40.925), (117.327, 40.94)],
    '滦平-承德': [(117.327, 40.94), (117.352, 40.95), (117.377, 40.968), (117.401, 40.987), (117.426, 40.99),
              (117.451, 40.99), (117.475, 40.993), (117.5, 40.984), (117.525, 40.993), (117.55, 40.99), (117.574, 40.984),
              (117.599, 40.972), (117.624, 40.956), (117.648, 40.953), (117.673, 40.953), (117.698, 40.953),
              (117.723, 40.953), (117.747, 40.953), (117.772, 40.953), (117.797, 40.965), (117.822, 40.981),
              (117.846, 40.984), (117.871, 40.99), (117.896, 40.99), (117.92, 40.984), (117.941, 40.965), (117.957, 40.95)],
    '承德-宽城': [(117.957, 40.95), (117.953, 40.931), (117.953, 40.912), (117.962, 40.894), (117.982, 40.875),
              (117.999, 40.856), (118.023, 40.856), (118.048, 40.838), (118.056, 40.819), (118.069, 40.8), (118.093, 40.8),
              (118.114, 40.785), (118.139, 40.775), (118.147, 40.76), (118.135, 40.741), (118.135, 40.722),
              (118.143, 40.707), (118.163, 40.7), (118.163, 40.682), (118.18, 40.669), (118.196, 40.65), (118.221, 40.647),
              (118.246, 40.654), (118.271, 40.657), (118.291, 40.644), (118.299, 40.625), (118.324, 40.616),
              (118.349, 40.6), (118.374, 40.597), (118.398, 40.597), (118.423, 40.594), (118.448, 40.594),
              (118.472, 40.604), (118.481, 40.61)],
    '喜峰口道': [(118.481, 40.61), (118.456, 40.594), (118.444, 40.575), (118.431, 40.557), (118.427, 40.538),
             (118.407, 40.522), (118.398, 40.503), (118.374, 40.485), (118.361, 40.466), (118.345, 40.447),
             (118.32, 40.434), (118.328, 40.416), (118.316, 40.397), (118.295, 40.378), (118.295, 40.359),
             (118.283, 40.343), (118.258, 40.334), (118.233, 40.321), (118.213, 40.306), (118.205, 40.287),
             (118.201, 40.268), (118.196, 40.249), (118.196, 40.23), (118.192, 40.215), (118.168, 40.202),
             (118.143, 40.196), (118.118, 40.186), (118.093, 40.183), (118.069, 40.18), (118.044, 40.18), (118.019, 40.18),
             (117.995, 40.18), (117.97, 40.189), (117.962, 40.189)],
    '罗文峪道': [(117.962, 40.189), (117.962, 40.208), (117.962, 40.227), (117.957, 40.246), (117.957, 40.265),
             (117.945, 40.284), (117.933, 40.303), (117.92, 40.321), (117.908, 40.34), (117.908, 40.359),
             (117.924, 40.369)] + _LUOWENYU_NORTH,
    '冷口道': [(118.699, 40.0), (118.716, 40.019), (118.736, 40.038), (118.753, 40.057), (118.769, 40.076),
            (118.786, 40.095), (118.794, 40.114), (118.798, 40.133), (118.819, 40.152), (118.835, 40.17), (118.835, 40.189),
            (118.819, 40.199), (118.819, 40.218), (118.81, 40.237), (118.823, 40.255), (118.847, 40.271), (118.851, 40.29),
            (118.864, 40.309), (118.88, 40.328), (118.884, 40.347), (118.909, 40.365), (118.913, 40.384),
            (118.938, 40.403), (118.942, 40.406)],
    '青龙-宽城': [(118.942, 40.406), (118.917, 40.416), (118.893, 40.409), (118.868, 40.412), (118.843, 40.406),
              (118.819, 40.406), (118.794, 40.4), (118.769, 40.403), (118.744, 40.4), (118.72, 40.416), (118.695, 40.419),
              (118.67, 40.422), (118.645, 40.425), (118.621, 40.434), (118.608, 40.453), (118.596, 40.472),
              (118.571, 40.488), (118.551, 40.497), (118.538, 40.516), (118.534, 40.535), (118.522, 40.553),
              (118.497, 40.572), (118.472, 40.588), (118.477, 40.607), (118.481, 40.61)],
    '界岭口道': [(119.239, 39.883), (119.235, 39.902), (119.218, 39.921), (119.206, 39.94), (119.202, 39.959),
             (119.202, 39.978), (119.206, 39.997), (119.226, 40.016), (119.243, 40.035), (119.247, 40.054),
             (119.251, 40.073), (119.272, 40.092), (119.296, 40.111), (119.296, 40.13), (119.292, 40.148), (119.28, 40.167),
             (119.259, 40.183), (119.243, 40.202), (119.23, 40.221), (119.206, 40.237), (119.181, 40.243),
             (119.16, 40.255), (119.144, 40.274), (119.136, 40.293), (119.123, 40.312), (119.099, 40.328),
             (119.074, 40.331), (119.057, 40.35), (119.041, 40.369), (119.02, 40.387), (118.996, 40.4), (118.971, 40.409),
             (118.946, 40.406), (118.942, 40.406)],
    '九门口道': [(119.622, 39.981), (119.647, 39.984), (119.671, 39.997), (119.696, 39.997), (119.721, 40.0),
             (119.745, 40.007), (119.77, 40.026), (119.787, 40.044), (119.795, 40.063), (119.795, 40.082),
             (119.783, 40.101), (119.758, 40.111), (119.737, 40.12)],
}
PASS_HALF_WIDTH = 2.6                       # cells (about 1.85 km each): five cells of canyon floor
PASS_WIDTH = {'九门口道': 2.2}
PASS_RIM = 1.0                              # the outer cells of a canyon may take cliff pieces
GATES = [  # where each road crosses the wall: (name, lon, lat, road); the gate itself stays open
    ('古北口', 117.157, 40.690, '古北口道'),
    ('喜峰口', 118.328, 40.420, '喜峰口道'),
    ('罗文峪', 117.947, 40.276, '罗文峪道'),
    ('冷口', 118.835, 40.190, '冷口道'),
    ('界岭口', 119.290, 40.160, '界岭口道'),
    ('九门口', 119.739, 40.120, '九门口道'),
]
GATE_OPEN = 1.3                             # half width of the opening in the sandbag line (cells)
WALL_HALF_WIDTH = 1.4                       # the wall ridge: mountain even where the contour dips (a river gorge)


def _wall():
    """The Ming wall from OSM: every mapped wall segment in the box."""
    path = os.path.join(osmwater.HERE, 'data/osm_changcheng/wall.json')
    return [[(p['lon'], p['lat']) for p in e['geometry']] for e in json.load(open(path))['elements']
            if e['type'] == 'way' and len(e.get('geometry', [])) > 1]


WALL = _wall()

STARTS = [  # 日军 / 中国军队
    ('承德', 117.957, 40.951),          # capital of 热河, given up without a fight on 4 March
    ('滦平', 117.327, 40.940),          # on the road down to 古北口
    ('宽城', 118.481, 40.610),          # behind 喜峰口: the 铃木 brigade's base
    ('青龙', 118.942, 40.406),          # 都山, behind 冷口 and 界岭口
    ('密云', 116.837, 40.375),          # 中央军 第17军 (关麟征, 黄杰, 刘戡) for 古北口 and 南天门
    ('遵化', 117.960, 40.188),          # 29军 宋哲元: 喜峰口 and 罗文峪
    ('迁安', 118.800, 40.030),          # 32军 商震: 冷口 (base east of the town, clear of the 滦河)
    ('秦皇岛·山海关', 119.600, 39.965),  # 何柱国 and 安德馨's battalion, 1-3 January
]
START_FLAT, START_ORE_DIST, START_WATER_CLEAR = 8, 7.5, 3
START_KEEP = 11                             # low ground kept around each start in the mountains (cells)
MAP_CENTER = (118.30, 40.35)
ORE_TOWNS = [  # 冀东 towns on the plain, and the villages along the pass roads (positions from OSM)
    ('平谷', 117.115, 40.139), ('蓟县', 117.401, 40.044), ('玉田', 117.732, 39.900), ('丰润', 118.156, 39.831),
    ('迁西', 118.290, 40.150), ('三屯营', 118.198, 40.230), ('沙河驿', 118.568, 39.866), ('古冶', 118.441, 39.733),
    ('卢龙', 118.876, 39.891), ('杨各庄', 118.866, 40.090), ('抚宁', 119.238, 39.882), ('昌黎', 119.161, 39.704),
    ('留守营', 119.299, 39.788), ('石匣', 117.059, 40.544), ('巴克什营', 117.192, 40.760), ('下板城', 118.140, 40.780),
    ('洒河桥', 118.258, 40.334), ('青龙-宽城道中', 118.538, 40.516), ('双滦', 117.790, 40.960), ('建昌营', 118.794, 40.114)]
GEM_TOWNS = [('南天门', 117.125, 40.616),        # 八道楼子 and 南天门, the last line south of 古北口 (Apr - May)
             ('半壁山', 117.924, 40.369),        # beyond 罗文峪
             ('喜峰口外 (潘家口)', 118.361, 40.466),   # 赵登禹's broadsword raid on the Japanese guns, 11-12 March
             ('九门口 (一片石)', 119.737, 40.120)]     # the wall across the 九江河, 'the first pass on water'
OIL = [('开滦 林西矿', 118.450, 39.760)]          # the 开滦 coal mines, which both sides wanted intact
TECH = [('CAOUTP', '天下第一关', 119.758, 40.004, 4, 3),
        ('CAAIRP', '承德机场', 118.020, 40.885, 3, 3)]
LANDMARKS = []
BRIDGES = [  # the 滦河 crossings of 1933, north to south
    ('迁西渡', 118.300, 40.115),
    ('迁安渡', 118.665, 40.000),
    ('滦州铁桥', 118.715, 39.750),     # the 北宁 railway bridge
]
CITIES = [('承德', 117.957, 40.951, 1.4), ('秦皇岛', 119.600, 39.935, 1.2), ('山海关', 119.760, 40.004, 0.9),
          ('遵化', 117.960, 40.188, 1.0), ('滦州', 118.697, 39.740, 1.0), ('密云', 116.837, 40.375, 0.9),
          ('迁安', 118.700, 40.000, 0.7), ('唐山', 118.180, 39.640, 1.0)]
CITY_COVER = 493                            # rammed earth and brick (temperate SAND), not modern pavement


def _rail(names, guide, step=0.02, tol=0.025):
    """A trunk line from the OSM ways with these names: the points near a hand-drawn guide, averaged by longitude."""
    path = os.path.join(osmwater.HERE, 'data/osm_changcheng/rail.json')
    P = np.array([(p['lon'], p['lat']) for e in json.load(open(path))['elements']
                  if e['tags'].get('name') in names and e['tags'].get('highspeed') != 'yes' for p in e['geometry']])
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


RAILWAYS = {  # the 北宁 railway (京奉), 唐山 - 滦州 - 昌黎 - 北戴河 - 秦皇岛 - 山海关 - 关外
    '北宁铁路': _rail({'京哈线', '津山线', '沈山线'},
                  [(118.10, 39.625), (118.20, 39.655), (118.44, 39.725), (118.70, 39.745), (119.00, 39.72), (119.16, 39.71),
                   (119.40, 39.83), (119.60, 39.94), (119.76, 40.00), (119.98, 40.12)]),
}
TRACKS = dict(PASSES, **{  # the roads of the plain
    '北平-遵化': [(116.837, 40.375), (117.0, 40.25), (117.115, 40.139), (117.401, 40.044), (117.60, 40.10),
              (117.960, 40.188)],
    '遵化-丰润-唐山': [(117.960, 40.188), (118.05, 40.02), (118.156, 39.831), (118.180, 39.640)],
    '遵化-迁西-迁安': [(117.962, 40.189), (118.19, 40.20), (118.290, 40.150), (118.45, 40.08), (118.665, 40.00),
                  (118.800, 40.030)],
    '迁安-卢龙-抚宁-山海关': [(118.800, 40.030), (118.876, 39.891), (119.05, 39.88), (119.238, 39.882),
                       (119.45, 39.93), (119.600, 39.965)],
    '卢龙-滦州-昌黎': [(118.876, 39.891), (118.75, 39.80), (118.697, 39.740), (118.95, 39.72), (119.161, 39.704)],
    '冷口-杨各庄': [(118.794, 40.114), (118.866, 40.090), (118.876, 39.891)],
})
TRACK_COVER = 'SAND'
ROADS = dict(RAILWAYS, **TRACKS)
RIVER_NAMES = set()
RIVER_BBOX = (116.0, 39.0, 121.0, 41.5)
WATER = 'data/osm_changcheng/water.json'
EXTRA_RIVERS = osmwater.lines(WATER, {'滦河', '潮河', '白河', '青龙河', '还乡河', '沙河', '瀑河', '柳河', '武烈河',
                                      '泃河', '黎河', '洒河', '伊逊河', '老牛河', '蚂蚁河'})
WATER_RIVERS = [[(x, y) for x, y in line if y < 40.26] for line in osmwater.lines(WATER, {'滦河'})]
WATER_RIVERS = [line for line in WATER_RIVERS if len(line) > 1]   # the 滦河 below 潘家口: out of the gorge to the sea
WATER_HALF_WIDTH = 0.8
LAKES = []
WATER_LEVEL = 0
MIN_WATER_CORNERS, MIN_LAND_CORNERS = 6, 8
BRIDGE_DETOUR, BRIDGE_MAX_LEN, BRIDGE_SEEDS = 30, 12, 80   # the 滦河 comes out of the cliffs: no way round on foot
MOUNTAIN_TREES = ['TREE25', 'TREE26', 'TREE27', 'TREE05', 'TREE06', 'TREE07', 'TREE08', 'TREE15']
VALLEY_TREES = ['TREE01', 'TREE02', 'TREE03', 'TREE04', 'TREE10', 'TREE11', 'TREE12', 'TREE17']
CHECK_POINTS = [(n, lo, la) for n, lo, la in GEM_TOWNS] + [(n, lo, la) for n, lo, la, _ in GATES]


def water_corners(g):
    """The 渤海 from the bathymetry."""
    return g.sample(g.box_blur(g.DEM, 1), g.CORNER_LON, g.CORNER_LAT) < -0.5


def city_cover(g, cover, city):
    """The wall walk: grey stone along the wall line wherever the plateau top is level."""
    cover[(g.polyline_distance(WALL) < 0.7) & g.high] = g.PAVE


def plateau(g):
    """(mountain mask, cells that must stay low, cells that must stay mountain). Low land is the 冀东 plain and the
    coast below the contour, the pass canyons and the towns along them; the wall is a ridge that stays mountain
    everywhere except where a road goes through a gate."""
    def grow(m, k=1):
        for _ in range(k):
            m = m | g.shift(m, 1, 0, False) | g.shift(m, -1, 0, False) | g.shift(m, 0, 1, False) | g.shift(m, 0, -1, False)
        return m

    passes = np.zeros_like(g.VALID)                   # the canyon floor ...
    core = np.zeros_like(g.VALID)                     # ... whose middle must stay low
    for name, line in PASSES.items():
        d = g.polyline_distance([line])
        passes |= d < PASS_WIDTH.get(name, PASS_HALF_WIDTH)
        core |= d < PASS_WIDTH.get(name, PASS_HALF_WIDTH) - PASS_RIM
    ridge = (g.polyline_distance(WALL) < WALL_HALF_WIDTH) & ~grow(passes, 2)   # room for cliff pieces at the gates
    sites = [(lon, lat, START_KEEP) for _, lon, lat in STARTS]
    sites += [(lon, lat, 3.5) for _, lon, lat in ORE_TOWNS] + [(lon, lat, 3) for _, lon, lat in GEM_TOWNS + OIL]
    sites += [(lon, lat, max(w, h) + 2) for _, _, lon, lat, w, h in TECH]
    sites += [(lon, lat, 6) for _, lon, lat in BRIDGES]                      # both banks of every crossing
    keep = core.copy()
    for lon, lat, r in sites:
        keep |= g.dist_to_point(lon, lat) < r
    keep &= ~grow(ridge, 2)                          # sites next to the wall give way to it, with room for the cliffs
    ridge &= ~grow(g.water_cell, 2)                  # where the 滦河 leaves the gorge
    lab, sizes = g.components(ridge)                 # wall fragments too small to carry cliff pieces are dropped
    ridge &= (lab >= 0) & (sizes[np.maximum(lab, 0)] >= 25)
    keep_all = keep | g.water_cell
    mount = (g.ELEV_CELL > PLATEAU_ELEV) & g.VALID
    mount = ~grow(~grow(mount, 2), 2)                 # close: valleys narrower than ~4 cells fill up
    mount = grow(~grow(~mount, 2), 2)                 # open: no ridges narrower than ~4 cells
    mount &= ~passes
    from cliffs import majority
    mount = (majority(majority(mount & ~keep_all, 2), 3) | ridge) & ~keep_all
    mount = (majority(mount, 3) | ridge) & ~keep_all   # once more after the ridge joins: no slivers along the wall
    low = g.VALID & ~mount & ~g.water_cell
    lab, sizes = g.components(low)
    bases = np.zeros_like(low)
    for _, lon, lat in STARTS:
        bases |= g.dist_to_point(lon, lat) < 3
    reach = np.unique(lab[(core | bases) & low])
    mount |= low & ~np.isin(lab, reach)               # valleys no army can walk to join the mountain
    lab, sizes = g.components(mount)
    mount &= (lab >= 0) & (sizes[np.maximum(lab, 0)] >= 60)
    firm = ridge & ~grow(keep_all | passes, 3)        # the cliff solver may reshape the wall's edges, not its middle
    return mount, keep_all, firm


def walls(g):
    """Sandbag works on the wall line across each gate, the gate itself left open: the cells of the wall polyline
    in the canyon of the gate's road."""
    near_wall = g.polyline_distance(WALL) < 0.8
    out = np.zeros_like(g.VALID)
    for _, lon, lat, road in GATES:
        canyon = g.polyline_distance([PASSES[road]]) < PASS_WIDTH.get(road, PASS_HALF_WIDTH) + 0.5
        gate = g.dist_to_point(lon, lat)
        line = near_wall & canyon & (gate < 7) & ~g.high & ~g.cliff_cell
        out |= line & (g.polyline_distance([PASSES[road]]) > GATE_OPEN)
    return out


def ground_cover(g):
    """March 1933: bare brown fields and dry grass on the plain, scrub and rock on the 燕山, pine on the high ridges."""
    E, R, lon, lat, n1, n2 = g.ELEV_CELL, g.RELIEF_CELL, g.CELL_LON, g.CELL_LAT, g.NOISE, g.NOISE2
    cover = np.full(E.shape, g.GRASS)
    plain = E < 220
    hills = (E >= 220) & (E < 900)
    high = E >= 900
    riparian = g.RIVER_D < 1.3
    coast = plain & (lon > 119.0) & (lat < 39.95)
    cover[plain & (n1 > 0.62)] = g.SAND                        # ploughed fields waiting for the spring sowing
    cover[plain & (n1 <= 0.62) & (n2 > 0.55)] = g.ROUGH        # dry winter grass
    cover[plain & (n1 < 0.25) & (n2 <= 0.55)] = g.THICK        # winter wheat
    cover[coast & (n2 > 0.6)] = g.SAND                         # the 北戴河 - 山海关 beaches
    cover[hills & (n1 > 0.45)] = g.ROUGH                       # rock and thorn scrub
    cover[hills & (n1 <= 0.45) & (n2 > 0.7)] = g.SAND
    cover[high & (n2 > 0.5)] = g.ROUGH
    p = np.full(E.shape, 0.008)
    p[riparian] = 0.06                                         # poplars along the 滦河 and the 潮河
    p[hills] = 0.05 + 0.08 * n2[hills]
    p[high] = 0.18 + 0.3 * n1[high]                            # pine and larch on 雾灵山 and 都山
    road = g.polyline_distance(PASSES.values())
    p[road < PASS_HALF_WIDTH + 1] *= 0.3                       # keep the passes open for armies
    p[road < 2.0] = 0
    mountain = hills | high
    return cover, mountain, p


# --- buildings ---------------------------------------------------------------------------
from regions.pools import POOLS as CIVILIAN_POOLS   # noqa: E402

CIVILIAN_START_CLEAR = 6
FLAT_CITIES = []
PROP_DENSITY = 0.2
CIVILIAN_ZONES = [  # lon, lat, radius (cells), pool, density
    (118.180, 39.650, 4, 'industrial', 0.5),     # 唐山 and the 开滦 mines
    (118.441, 39.745, 3, 'industrial', 0.4),     # 古冶 / 林西
    (119.600, 39.930, 4, 'village', 0.4),        # 秦皇岛 harbour
    (119.487, 39.822, 3, 'houses', 0.5),         # 北戴河 villas
    (118.697, 39.740, 4, 'village', 0.35),       # 滦州
    (117.401, 40.044, 4, 'village', 0.35),       # 蓟县
    (117.115, 40.139, 4, 'village', 0.3),        # 平谷
    (119.161, 39.704, 4, 'village', 0.3),        # 昌黎
    (117.700, 39.950, 6, 'farm', 0.2),           # 玉田
    (118.400, 39.950, 6, 'farm', 0.2),
    (118.950, 39.800, 5, 'farm', 0.2),
    (117.300, 40.150, 5, 'farm', 0.2),
]
CIVILIAN_FILL = ('village', 0.5)
