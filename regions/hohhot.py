"""Hohhot (呼和浩特): 8 players around a neutral city, Daqing Shan, Tumed plain, loess hills, Yellow River."""
import json
import os
from collections import deque

import numpy as np

NAME = 'Hohhot'
OUTFILE = 'hohhot.mpr'
SEED = 1581  # Guihua city founded by Altan Khan
W, H = 117, 136
LX, LY, LW, LH = 2, 4, 113, 123
LON0, LON1, LAT0, LAT1 = 110.62, 112.41, 41.35, 39.85
DEM_DIR, DEM_ZOOM = 'data/dem_hohhot', 9
BLUR_SMOOTH, BLUR_WIDE, BLUR_TERRAIN = 2, 8, 4
ELEV_PTS = [980, 1090, 1190, 1310, 1450, 1600, 1760, 1950, 2300]
LVL_PTS = [0, 0, 1, 2, 3, 4, 5.2, 6.6, 8.6]
LIGHTING = dict(Red=1.0, Green=0.97, Blue=0.92, Level=0.03, Ambient=1.0)

STARTS = [
    ('武川', 111.46, 41.09),
    ('西乌兰不浪', 110.92, 41.18),
    ('察素齐', 111.15, 40.70),
    ('托克托', 111.31, 40.37),
    ('和林格尔', 111.83, 40.38),
    ('清水河', 111.68, 39.98),
    ('赛罕东', 112.18, 40.63),
    ('武川东', 112.12, 41.15),
]
START_FLAT, START_ORE_DIST = 11, 8.0
MAP_CENTER = (111.68, 40.70)
ORE_TOWNS = [('毕克齐', 111.38, 40.75), ('盛乐', 111.72, 40.56), ('喇嘛湾', 111.48, 40.12),
             ('蜈蚣坝', 111.62, 40.97), ('榆林', 112.00, 40.86), ('哈乐', 111.80, 41.20),
             ('五申', 111.20, 41.03), ('新店子', 112.10, 40.25), ('哈素海北', 111.02, 40.72),
             ('大黑河', 111.48, 40.55)]
GEM_TOWNS = [('准格尔煤田', 111.02, 40.02), ('准格尔北', 110.82, 40.22), ('大青山', 111.88, 40.97)]
OIL = [('托克托电厂', 111.45, 40.25), ('准格尔', 110.95, 39.92), ('金山', 111.52, 40.66),
       ('可镇北', 111.66, 41.17), ('和林东', 112.02, 40.45), ('清水河东', 111.95, 40.03)]
TECH = [('CAAIRP', '白塔机场', 111.82, 40.85, 3, 3),
        ('CAHOSP', '内蒙古医院', 111.57, 40.84, 6, 4),
        ('CAOUTP', '昭君博物院', 111.66, 40.69, 4, 3)]
CITIES = [('呼和浩特', 111.70, 40.81, 8), ('察素齐', 111.15, 40.70, 2.5), ('武川', 111.46, 41.09, 2.5),
          ('托克托', 111.31, 40.37, 2.5), ('和林格尔', 111.83, 40.38, 2.5), ('清水河', 111.68, 39.98, 2),
          ('毕克齐', 111.38, 40.75, 1.5)]
ROADS = {
    'G6 west': [(111.68, 40.82), (111.38, 40.73), (111.15, 40.69), (110.55, 40.62)],
    'Hohhot-Wuchuan': [(111.68, 40.82), (111.62, 40.95), (111.46, 41.09), (110.92, 41.18)],
    'G6 east': [(111.68, 40.82), (112.00, 40.85), (112.50, 40.90)],
    'South': [(111.68, 40.82), (111.83, 40.38), (111.68, 39.98)],
    'Togtoh': [(111.68, 40.82), (111.31, 40.37)],
    'Wuchuan east': [(111.46, 41.09), (112.12, 41.15)],
    'East': [(111.68, 40.82), (112.18, 40.63)],
}
RIVER_NAMES = {'Huang'}
RIVER_BBOX = (110.2, 39.4, 112.8, 41.8)
EXTRA_RIVERS = [  # traced by hand: 大黑河, 小黑河, 浑河
    [(112.30, 40.98), (112.05, 40.88), (111.90, 40.78), (111.75, 40.72), (111.58, 40.63),
     (111.42, 40.50), (111.31, 40.37), (111.22, 40.27)],
    [(111.85, 40.95), (111.70, 40.78), (111.55, 40.70), (111.42, 40.58)],
    [(112.30, 40.15), (111.95, 40.10), (111.70, 40.05), (111.50, 40.00), (111.38, 39.97)],
]


def _yellow_river():
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'rivers.geojson')
    lines = []
    for ft in json.load(open(path))['features']:
        geom = ft['geometry']
        if ft['properties'].get('name') != 'Huang' or not geom:
            continue
        for line in (geom['coordinates'] if geom['type'] == 'MultiLineString' else [geom['coordinates']]):
            if any(110.0 <= x <= 113.0 and 39.3 <= y <= 41.8 for x, y in line):
                lines.append(line)
    return lines


WATER_RIVERS = _yellow_river()
WATER_HALF_WIDTH = 2.4
LAKES = [(110.97, 40.60, 5.0, 3.0)]   # 哈素海
WATER_LEVEL = 0
MOUNTAIN_TREES = ['TREE25', 'TREE26', 'TREE27', 'TREE05', 'TREE06', 'TREE07', 'TREE08', 'TREE15']
VALLEY_TREES = ['TREE01', 'TREE02', 'TREE03', 'TREE04', 'TREE10', 'TREE11', 'TREE12', 'TREE17']


def far_bank(g):
    """Land not reachable from the city without crossing water (the Ordos side of the Yellow River)."""
    sx, sy = (int(v) for v in g.to_iso(111.68, 40.82))
    seen = np.zeros(g.VALID.shape, bool)
    q = deque([(sx, sy)])
    seen[sx, sy] = True
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < g.N and 0 <= ny < g.N and not seen[nx, ny] and g.VALID[nx, ny] and not g.water_cell[nx, ny]:
                seen[nx, ny] = True
                q.append((nx, ny))
    return g.VALID & ~seen & ~g.water_cell


def ground_cover(g):
    E, R, lon, lat, n1, n2 = g.ELEV_CELL, g.RELIEF_CELL, g.CELL_LON, g.CELL_LAT, g.NOISE, g.NOISE2
    cover = np.full(E.shape, g.GRASS)
    p = np.full(E.shape, 0.01)
    riparian = (g.RIVER_D < 1.3) & (E < 1250)
    across = far_bank(g)
    mountains = (lat > 40.82) & (lat < 41.08) & ((E > 1300) | (R > 40))
    plateau = (lat >= 41.0) & (E > 1450) & ~mountains
    loess = (lat < 40.58) & (E > 1150) & ~across
    plain = (E <= 1150) & ~across

    cover[plain & (n1 > 0.62)] = g.THICK                       # irrigated fields of the Tumed plain
    cover[plain & (n2 > 0.76) & (n1 <= 0.62)] = g.ROUGH        # saline flats
    p[plain] = 0.015

    cover[plateau & (n2 > 0.6)] = g.ROUGH                      # dry steppe
    cover[plateau & (n1 > 0.66)] = g.THICK
    p[plateau] = 0.008

    cover[mountains & (n1 > 0.45)] = g.ROUGH                   # bare rock of Daqing Shan
    cover[mountains & (n1 < 0.32)] = g.THICK                   # forested north slopes
    p[mountains] = 0.03 + 0.17 * (1 - n1[mountains])

    cover[loess & (n1 > 0.35)] = g.SAND                        # loess hills of Horinger and Qingshuihe
    cover[loess & (n1 <= 0.35) & (n2 > 0.6)] = g.ROUGH
    p[loess] = 0.02

    cover[across & (n1 > 0.3)] = g.SAND                        # Jungar side of the Yellow River
    cover[across & (n1 <= 0.3) & (n2 > 0.5)] = g.ROUGH
    p[across] = 0.01

    cover[riparian] = g.THICK
    p[riparian] = 0.3
    cover[E > 1950] = g.ROUGH
    return cover, (E > 1300), p

# --- buildings ---------------------------------------------------------------------------
from regions.pools import POOLS as CIVILIAN_POOLS   # noqa: E402

CIVILIAN_START_CLEAR = 10
FLAT_CITIES = [('呼和浩特', 111.70, 40.81, 8)]
PROP_DENSITY = 0.3
CIVILIAN_ZONES = [  # lon, lat, radius (cells), pool, density
    (111.67, 40.82, 4, 'highrise', 0.9),         # 新城/回民区 city centre
    (111.75, 40.80, 3, 'highrise', 0.8),         # 赛罕区 new district
    (111.58, 40.75, 3, 'industrial', 0.6),       # 金山开发区
    (111.45, 40.25, 3, 'industrial', 0.6),       # 托克托电厂
    (111.35, 40.55, 5, 'farm', 0.25),            # 土默特 plain villages
    (111.55, 40.45, 4, 'farm', 0.25),
    (111.95, 40.70, 4, 'farm', 0.25),
    (111.25, 40.80, 3, 'village', 0.4),
    (111.60, 41.25, 5, 'farm', 0.2),             # 武川 plateau
    (111.20, 41.12, 4, 'village', 0.3),
    (111.95, 40.15, 3, 'village', 0.4),          # loess villages
    (111.50, 40.18, 3, 'village', 0.4),
    (112.20, 40.40, 3, 'village', 0.4),
]
CIVILIAN_FILL = ('small', 0.8)
