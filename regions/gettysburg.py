"""Battle of Gettysburg (1-3 July 1863): ten roads meet at a small Pennsylvania town, and two armies marching on
them collided there. The fishhook (Culp's Hill - Cemetery Hill - Cemetery Ridge - the Round Tops) faces Seminary
Ridge across the fields of Pickett's Charge; Oak Hill and McPherson Ridge in the north-west, South Mountain and the
Cashtown Gap in the west. Roads, creeks and the railroad come from OpenStreetMap (data/osm_gettysburg)."""
import json
import os

import numpy as np

from regions import osmwater

NAME = 'Battle of Gettysburg'
OUTFILE = 'gettysburg.mpr'
SEED = 1863  # 1-3 July 1863
W, H = 220, 192
LX, LY, LW, LH = 2, 4, 216, 179
LON0, LON1, LAT0, LAT1 = -77.385, -77.15, 39.905, 39.7546    # 66 m a cell
DEM_DIR, DEM_ZOOM = 'data/dem_gettysburg', 13
BLUR_SMOOTH, BLUR_WIDE, BLUR_TERRAIN = 1, 10, 1
# Only ~40 m between Rock Creek and the hilltops, and Culp's Hill or Little Round Top are a handful of cells wide at
# 66 m a cell: as they are they would come out as one-level bumps. So the DEM is prepared with the local relief of
# the battle's hills multiplied (HILLS: each hill keeps its real shape, only taller), and the rest of the map gets a
# gentle scale: the broad Seminary and Cemetery ridges a level above the fields, the farmland open and deployable,
# South Mountain high in the west.
ELEV_PTS = [0, 150, 172, 195, 220, 260, 320, 400, 500]
LVL_PTS = [0, 0, 1, 2, 3, 4.5, 6.5, 8.5, 10.5]
RELIEF_RADIUS = 14      # DEM pixels (14.6 m at zoom 13) per box-blur pass: the "surroundings" a hill stands above
HILLS = [  # name, lon, lat, radius (km), relief gain
    ("Culp's Hill", -77.2200, 39.8190, 0.40, 4.5),
    ('Cemetery Hill', -77.2295, 39.8218, 0.30, 5.0),
    ('Little Round Top', -77.2366, 39.7922, 0.28, 4.0),
    ('Big Round Top', -77.2394, 39.7867, 0.40, 3.0),
    ('Oak Hill', -77.2426, 39.8499, 0.35, 3.0),
    ('Powers Hill', -77.2211, 39.8060, 0.25, 3.0),
    ("Wolf's Hill", -77.2080, 39.8160, 0.30, 2.5),
    ("Benner's Hill", -77.2143, 39.8260, 0.25, 1.5),
]


def _prepare_dem():
    """data/dem_gettysburg/raw/*.png (AWS terrarium tiles) -> data/dem_gettysburg/*.png with the HILLS raised."""
    import glob
    import math
    from PIL import Image
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', DEM_DIR)
    tiles = sorted(glob.glob(os.path.join(here, 'raw', f'{DEM_ZOOM}_*.png')))
    stamp = repr((RELIEF_RADIUS, HILLS))
    stamp_file = os.path.join(here, 'prepared.txt')
    if tiles and os.path.exists(stamp_file) and open(stamp_file).read() == stamp:
        return
    key = [tuple(int(v) for v in os.path.basename(t)[:-4].split('_')[1:]) for t in tiles]
    tx0, ty0 = min(k[0] for k in key), min(k[1] for k in key)
    E = np.zeros((256 * (max(k[1] for k in key) - ty0 + 1), 256 * (max(k[0] for k in key) - tx0 + 1)))
    for t, (x, y) in zip(tiles, key):
        a = np.asarray(Image.open(t).convert('RGB'), float)
        E[(y - ty0) * 256:(y - ty0 + 1) * 256, (x - tx0) * 256:(x - tx0 + 1) * 256] = \
            a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768
    B, r = E.copy(), RELIEF_RADIUS
    for _ in range(3):                                   # the same box blur as gen.py
        for axis in (0, 1):
            p = np.pad(B, [(r + 1, r) if ax == axis else (0, 0) for ax in (0, 1)], mode='edge')
            c = np.cumsum(p, axis=axis)
            B = (np.take(c, range(2 * r + 1, c.shape[axis]), axis=axis) -
                 np.take(c, range(0, c.shape[axis] - 2 * r - 1), axis=axis)) / (2 * r + 1)
    n = 256 * 2 ** DEM_ZOOM
    lon = (tx0 * 256 + np.arange(E.shape[1]) + 0.5) / n * 360 - 180
    lat = np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * (ty0 * 256 + np.arange(E.shape[0]) + 0.5) / n))))
    LON, LAT = np.meshgrid(lon, lat)
    gain = np.zeros(E.shape)
    for _, hl, ha, rad, k in HILLS:
        d = np.hypot((LON - hl) * 111.32 * math.cos(math.radians(ha)), (LAT - ha) * 111.32)
        gain = np.maximum(gain, k * np.clip((1.5 * rad - d) / (0.5 * rad), 0, 1))
    E = E + gain * np.maximum(E - B, 0)
    v = np.round((E + 32768) * 256).astype(np.int64)
    rgb = np.stack([v >> 16, (v >> 8) & 255, v & 255], axis=-1).astype(np.uint8)
    for t, (x, y) in zip(tiles, key):
        Image.fromarray(rgb[(y - ty0) * 256:(y - ty0 + 1) * 256, (x - tx0) * 256:(x - tx0 + 1) * 256]).save(
            os.path.join(here, os.path.basename(t)))
    open(stamp_file, 'w').write(stamp)


_prepare_dem()
LIGHTING = dict(Red=1.03, Green=1.0, Blue=0.94, Level=0.03, Ambient=1.04)   # high summer, 87 F on 2 July

STARTS = [  # Confederates from the west and north, the Union behind the fishhook and on its approaches.
    # A base levels the ground around it, so the bases sit beside the famous hills, which stay hills.
    ('Cashtown 卡什敦', -77.3450, 39.8780),                     # A.P. Hill concentrates below the Cashtown Gap
    ('Oak Hill 橡树山', -77.2480, 39.8600),                     # Rodes' guns open from Oak Hill on 1 July
    ('Heidlersburg Road 海德斯堡路', -77.2060, 39.8700),        # Ewell's corps comes down from the north
    ("Culp's Hill 卡尔普山", -77.2040, 39.8150),                 # the barb of the fishhook, held from across Rock Creek
    ('East Cavalry Field 东骑兵战场', -77.1661, 39.8329),        # Stuart against Custer, 3 July
    ('Two Taverns 两酒馆', -77.1655, 39.7784),                  # Slocum's XII Corps on the Baltimore Pike
    ('Round Tops 圆顶山', -77.2240, 39.7940),                   # behind Little Round Top: Chamberlain and the 20th Maine
    ('Seminary Ridge 神学院岭', -77.2509, 39.8119),             # Spangler's Woods, where Pickett's Charge stepped off
]
START_FLAT, START_ORE_DIST, START_WATER_CLEAR = 9, 6.5, 3
MAP_CENTER = (-77.236, 39.815)
ORE_TOWNS = [('Mummasburg', -77.2894, 39.8856), ('Hunterstown', -77.1603, 39.8823), ('Bonneauville', -77.1378, 39.8128),
             ('Fairfield', -77.3686, 39.7873), ('Greenmount', -77.2736, 39.7643), ('Barlow', -77.2300, 39.7554),
             ('Black Horse Tavern', -77.2828, 39.8202), ('Knoxlyn', -77.3169, 39.8308), ('McKnightstown', -77.3263, 39.8711),
             ('Goldenville', -77.2419, 39.8926), ('Seven Stars', -77.2967, 39.8583), ('Germantown', -77.1474, 39.7695),
             ('Shrivers Corners', -77.1889, 39.8926), ('Hammer Hall', -77.3033, 39.8395), ('Orrtanna', -77.3608, 39.8451),
             ('Virginia Mills', -77.3761, 39.8029), ('Lake Heritage', -77.1839, 39.8095), ('Herr Ridge', -77.2700, 39.8370),
             ('Pitzer Farm', -77.2700, 39.8020), ('Bushman Farm', -77.2560, 39.7790), ('McAllister Mill', -77.2050, 39.8030),
             ('Guldens', -77.1389, 39.8584), ('Benner Farm', -77.2050, 39.8420)]
GEM_TOWNS = [('High Water Mark 高水位线', -77.2360, 39.8129),   # the copse of trees Pickett's men aimed for
             ("Devil's Den 魔鬼穴", -77.2426, 39.7914),           # the boulder field below Little Round Top
             ("Benner's Hill 本纳山", -77.2143, 39.8260),         # Ewell's guns against Culp's Hill
             ('Sugar Loaf 糖块山', -77.3533, 39.8076),
             ('Granite Hill 花岗岩山', -77.1605, 39.8540)]
OIL = []
TECH = [('CAHOSP', 'Pennsylvania College 宾夕法尼亚学院 (field hospital)', -77.2365, 39.8365, 6, 4),
        ('CAOUTP', 'Lutheran Seminary 路德神学院 (Buford watched from its cupola)', -77.2445, 39.8318, 4, 3)]
CITIES = [('Gettysburg 葛底斯堡', -77.2310, 39.8312, 6)]
CITY_COVER = 493          # dirt streets in 1863 (temperate SAND)


_OSM = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'osm_gettysburg', 'all.json')
_WAYS = json.load(open(_OSM))['elements']


def _lines(names, key='highway', near=None):
    """Polylines of the OSM ways of these names (optionally only those with a node within `near` of a point)."""
    out = []
    for e in _WAYS:
        t = e.get('tags', {})
        if e['type'] == 'way' and key in t and t.get('name') in names:
            pts = [(p['lon'], p['lat']) for p in e['geometry']]
            if near and min(abs(x - near[0]) + abs(y - near[1]) for x, y in pts) > near[2]:
                continue
            out.append(pts)
    return out


_ROADS_1863 = {  # the ten roads into town, on their modern alignments
    'Chambersburg Pike': {'Chambersburg Road', 'Chambersburg Street', 'Buford Avenue', 'US 30', 'Lincoln Square'},
    'Mummasburg Road': {'Mummasburg Road'},
    'Carlisle Road': {'Carlisle Street', 'Biglerville Road'},
    'Harrisburg Road': {'Harrisburg Street', 'Old Harrisburg Road'},
    'York Pike': {'York Street', 'York Road'},
    'Hanover Road': {'Hanover Road', 'East Hanover Street'},
    'Baltimore Pike': {'Baltimore Street', 'Baltimore Pike'},
    'Taneytown Road': {'Taneytown Road'},
    'Emmitsburg Road': {'Steinwehr Avenue', 'Emmitsburg Road'},
    'Fairfield Road': {'West Middle Street', 'Fairfield Road'},
    'Wheatfield Road': {'Wheatfield Road', 'Millerstown Road'},
    'Hunterstown Road': {'Hunterstown Road'},
}
TRACKS = {}
for _name, _osm in _ROADS_1863.items():
    for _i, _l in enumerate(_lines(_osm)):
        TRACKS[f'{_name} {_i}'] = _l
# the unfinished railroad grade west of town (the Railroad Cut of 1 July): graded, no rails yet
for _i, _l in enumerate(l for l in _lines({'Hanover Subdivision'}, 'railway') if min(x for x, _ in l) > -77.29
                        and max(x for x, _ in l) <= -77.231):
    TRACKS[f'Railroad Cut {_i}'] = _l
TRACK_COVER = 'SAND'
RAILWAYS = {f'Gettysburg Railroad {i}': l for i, l in enumerate(_lines({'Hanover Subdivision'}, 'railway'))
            if min(x for x, _ in l) >= -77.2312}   # 1858 line east to Hanover
ROADS = dict(TRACKS, **RAILWAYS)

RIVER_NAMES = set()
RIVER_BBOX = (-77.45, 39.70, -77.08, 39.95)
EXTRA_RIVERS = osmwater.lines('data/osm_gettysburg/all.json',
                              {'Willoughby Run', 'Marsh Creek', 'Rock Creek', 'Plum Run', 'Stevens Run', 'White Run',
                               'Spangler Spring Run', 'Pitzer Run', 'Rose Run', 'Middle Creek', 'Littles Run',
                               "Little's Run", 'Conewago Creek', 'Muddy Run', 'Swift Run', 'Cress Run', 'Alloway'})
WATER_RIVERS, WATER_HALF_WIDTH, LAKES = [], 0, []
WATER_LEVEL = 0
MIN_WATER_CORNERS, MIN_LAND_CORNERS = 6, 8
MOUNTAIN_TREES = ['TREE05', 'TREE06', 'TREE07', 'TREE08', 'TREE15', 'TREE01', 'TREE02', 'TREE03']   # oak-hickory woods
VALLEY_TREES = ['TREE01', 'TREE02', 'TREE03', 'TREE04', 'TREE10', 'TREE11', 'TREE12', 'TREE17']

WOODS = [  # woods of July 1863: lon, lat, radius (km)
    (-77.2190, 39.8180, 0.55),   # Culp's Hill
    (-77.2394, 39.7860, 0.60),   # Big Round Top
    (-77.2440, 39.7955, 0.30),   # Rose Woods / Stony Hill
    (-77.2560, 39.8350, 0.25),   # McPherson's (Herbst) Woods
    (-77.2520, 39.8120, 0.30),   # Spangler's Woods
    (-77.2590, 39.8040, 0.35),   # Pitzer's Woods
    (-77.2520, 39.8230, 0.25),   # Seminary Ridge woods
    (-77.2080, 39.8160, 0.40),   # Wolf's Hill
    (-77.2130, 39.8060, 0.30),   # Powers Hill / McAllister's woods
    (-77.1690, 39.8370, 0.35),   # Rummel woods, East Cavalry Field
]
FIELDS = [  # open farmland the fighting is named after: lon, lat, radius (km)
    (-77.2479, 39.7997, 0.25),   # the Peach Orchard
    (-77.2424, 39.7976, 0.20),   # the Wheatfield
    (-77.2440, 39.8130, 0.55),   # the fields of Pickett's Charge
    (-77.2400, 39.7930, 0.12),   # the Valley of Death
]


def _km(g, lon, lat):
    return np.hypot((g.CELL_LON - lon) * 85.4, (g.CELL_LAT - lat) * 111.3)


def ground_cover(g):
    E, R, lon, lat, n1, n2 = g.ELEV_CELL, g.RELIEF_CELL, g.CELL_LON, g.CELL_LAT, g.NOISE, g.NOISE2
    cover = np.full(E.shape, g.GRASS)
    p = np.full(E.shape, 0.012)
    mountain = E > 240                                            # South Mountain
    knob = (R > 9) & ~mountain                                    # wooded, boulder-strewn knobs and ridge crests
    riparian = g.RIVER_D < 1.3
    woods = np.zeros(E.shape, bool)
    for wl, wa, r in WOODS:
        woods |= _km(g, wl, wa) < r * (0.75 + 0.5 * n2)
    fields = np.zeros(E.shape, bool)
    for fl, fa, r in FIELDS:
        fields |= _km(g, fl, fa) < r

    cover[n1 > 0.5] = g.THICK                                     # ripe wheat and hay
    cover[(n1 <= 0.5) & (n2 > 0.7)] = g.SAND                      # ploughed ground, farm lanes
    cover[knob & (n2 > 0.4)] = g.ROUGH
    cover[_km(g, -77.2426, 39.7914) < 0.22] = g.ROUGH             # Devil's Den boulders
    cover[_km(g, -77.2357, 39.7911) < 0.15] = g.ROUGH             # Little Round Top's rocky west face
    cover[fields] = g.THICK
    p[riparian] = 0.18
    p[knob] = 0.15
    p[woods] = 0.45
    p[mountain] = 0.30 + 0.3 * (1 - n1[mountain])
    p[fields] = 0.0
    p[_km(g, -77.2357, 39.7911) < 0.25] = 0.02                   # Little Round Top was cleared of timber
    return cover, mountain | woods, p


# --- buildings ---------------------------------------------------------------------------
from regions.pools import POOLS as CIVILIAN_POOLS   # noqa: E402

CIVILIAN_START_CLEAR = 6
PROP_DENSITY = 0.25
CIVILIAN_ZONES = [  # lon, lat, radius (cells), pool, density
    (-77.2310, 39.8312, 6, 'village', 0.7),       # Gettysburg: brick houses of 2400 people
    (-77.2479, 39.8005, 2, 'farm', 0.6),          # Sherfy farm and the Peach Orchard
    (-77.2405, 39.8090, 1.5, 'farm', 0.6),        # Codori farm on the Emmitsburg Road
    (-77.2493, 39.7956, 1.5, 'farm', 0.6),        # Rose farm
    (-77.2535, 39.8370, 1.5, 'farm', 0.6),        # McPherson farm
    (-77.2370, 39.8020, 1.5, 'farm', 0.5),        # Trostle farm
    (-77.1705, 39.8326, 1.5, 'farm', 0.6),        # Rummel farm
    (-77.2828, 39.8202, 1.5, 'village', 0.6),     # Black Horse Tavern
    (-77.3594, 39.8845, 2, 'village', 0.5),       # Cashtown inn
    (-77.3686, 39.7873, 2.5, 'village', 0.5),     # Fairfield
    (-77.1655, 39.7784, 2, 'village', 0.4),       # Two Taverns
    (-77.1378, 39.8128, 2, 'village', 0.4),       # Bonneauville
    (-77.1603, 39.8823, 2, 'village', 0.5),       # Hunterstown
    (-77.2894, 39.8856, 2, 'village', 0.5),       # Mummasburg
    (-77.2600, 39.8600, 5, 'farm', 0.2),
    (-77.2900, 39.8000, 5, 'farm', 0.2),
    (-77.1900, 39.7900, 5, 'farm', 0.2),
    (-77.2000, 39.8600, 5, 'farm', 0.2),
]
CIVILIAN_FILL = ('small', 0.6)
