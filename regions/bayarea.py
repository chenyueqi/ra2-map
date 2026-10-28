"""San Francisco Bay Area (URBAN theater): 8 players around the bay, water from bathymetry, city blocks, Marin cut off."""
import numpy as np

NAME = 'San Francisco Bay'
OUTFILE = 'bayarea.mpr'
THEATER = 'URBAN'
SEED = 1849  # gold rush
W, H = 100, 239
LX, LY, LW, LH = 2, 4, 96, 226
LON0, LON1, LAT0, LAT1 = -122.66, -121.70, 38.10, 37.20
DEM_DIR, DEM_ZOOM = 'data/dem_bayarea', 10
BLUR_SMOOTH, BLUR_WIDE, BLUR_TERRAIN = 2, 7, 3
ELEV_PTS = [0, 25, 70, 140, 230, 340, 470, 620, 800, 1100]
LVL_PTS = [0, 0, 1, 2, 3, 4.2, 5.4, 6.6, 7.8, 9.4]
LIGHTING = dict(Red=0.95, Green=0.97, Blue=1.02, Level=0.03, Ambient=1.02)

STARTS = [
    ('San Francisco', -122.445, 37.755),
    ('Oakland', -122.21, 37.80),
    ('Walnut Creek', -122.05, 37.91),
    ('Livermore', -121.77, 37.68),
    ('San Jose', -121.88, 37.33),
    ('Fremont', -121.96, 37.55),
    ('Palo Alto', -122.17, 37.42),
    ('Half Moon Bay', -122.395, 37.475),
]
START_FLAT, START_ORE_DIST, START_WATER_CLEAR = 11, 8.5, 3
MAP_CENTER = (-122.15, 37.65)
ORE_TOWNS = [('San Bruno Mountain', -122.43, 37.685), ('Belmont hills', -122.31, 37.52), ('Redwood City', -122.24, 37.49),
             ('Shoreline', -122.08, 37.425), ('Milpitas', -121.90, 37.43), ('Dublin', -121.93, 37.70),
             ('Tilden Park', -122.24, 37.895), ('Concord', -122.03, 37.98), ('San Leandro', -122.15, 37.72),
             ('Danville', -121.99, 37.82), ('Cupertino', -122.03, 37.32)]
GEM_TOWNS = [('Mount Diablo', -121.92, 37.87), ('Mount Tamalpais', -122.59, 37.92),
             ('San Rafael', -122.53, 37.98), ('Sausalito', -122.49, 37.87)]
ORE_DRILLS = [GEM_TOWNS[0]]
OIL = [('Richmond refinery', -122.38, 37.945), ('Rodeo refinery', -122.26, 38.03),
       ('Martinez refinery', -122.11, 38.01), ('Hunters Point', -122.375, 37.725),
       ('Moffett Field', -122.05, 37.41), ('Alcatraz', -122.423, 37.827)]
TECH = [('CAAIRP', 'SFO', -122.385, 37.62, 3, 3),
        ('CAHOSP', 'Hayward', -122.08, 37.665, 6, 4),
        ('CAOUTP', 'Marin Headlands', -122.50, 37.86, 4, 3)]
ISLANDS = [(-122.423, 37.827, 3.8), (-122.428, 37.857, 3.2)]   # Alcatraz, Angel Island (too small for the DEM at this scale)
MIN_WATER_CORNERS, MIN_LAND_CORNERS = 6, 6
CITIES = [
    ('San Francisco', -122.43, 37.765, 9.2), ('Oakland', -122.25, 37.8, 7.2), ('Berkeley', -122.27, 37.87, 3.9),
    ('San Jose', -121.89, 37.33, 8.5), ('Fremont', -121.96, 37.55, 4.6), ('Palo Alto', -122.15, 37.43, 3.9),
    ('San Mateo', -122.32, 37.56, 3.9), ('Walnut Creek', -122.05, 37.91, 3.9), ('Livermore', -121.77, 37.68, 3.9),
    ('Hayward', -122.08, 37.665, 3.9), ('Richmond', -122.35, 37.94, 3.9), ('Daly City', -122.47, 37.69, 3.3),
    ('Sunnyvale', -122.03, 37.38, 5.2), ('Santa Clara', -121.96, 37.35, 3.9), ('Concord', -122.03, 37.98, 3.9),
    ('San Rafael', -122.53, 37.97, 3.3), ('Vallejo', -122.25, 38.1, 3.3), ('Half Moon Bay', -122.42, 37.47, 2.6),
]
ROADS = {
    'I-280': [(-122.45, 37.73), (-122.33, 37.53), (-122.17, 37.40), (-121.90, 37.33)],
    'CA-24': [(-122.24, 37.84), (-122.06, 37.90)],
    'I-580': [(-122.13, 37.72), (-121.93, 37.70), (-121.77, 37.68), (-121.55, 37.75)],
    'I-680': [(-122.06, 37.90), (-121.93, 37.70), (-121.90, 37.45), (-121.88, 37.33)],
    'CA-92': [(-122.32, 37.55), (-122.42, 37.47)],
    'CA-17': [(-121.95, 37.30), (-122.00, 37.10)],
    'I-80': [(-122.27, 37.83), (-122.35, 37.95), (-122.26, 38.03)],
}
RIVER_NAMES = set()
RIVER_BBOX = (-123.0, 36.9, -121.4, 38.4)
EXTRA_RIVERS = []
WATER_RIVERS, WATER_HALF_WIDTH, LAKES = [], 0, []
WATER_LEVEL = 0
MOUNTAIN_TREES = ['TREE25', 'TREE26', 'TREE27', 'TREE05', 'TREE06', 'TREE07', 'TREE08', 'TREE15']
VALLEY_TREES = ['TREE01', 'TREE02', 'TREE03', 'TREE04', 'TREE10', 'TREE11', 'TREE12', 'TREE17']


def water_corners(g):
    """Pacific and the bay straight from the bathymetry in the elevation tiles."""
    return g.sample(g.box_blur(g.DEM, 1), g.CORNER_LON, g.CORNER_LAT) < -0.5


# URBAN ground: 0 concrete, 493 grass, 131 dirt, 418 gravel, 534 brick plaza
CITY_COVER = 0
PARKS = [(-122.484, 37.769, 4.2, 1.0), (-122.466, 37.798, 2.2, 1.4)]   # Golden Gate Park, Presidio (km half-axes)
DOWNTOWNS = [
    (-122.4, 37.79, 3.3), (-122.27, 37.805, 2.6), (-121.89, 37.335, 2.6),
]


def _parks(g):
    m = np.zeros(g.CELL_LON.shape, bool)
    for lon, lat, a, b in PARKS:
        dx = (g.CELL_LON - lon) * 111.32 * np.cos(np.radians(lat)) / a
        dy = (g.CELL_LAT - lat) * 111.0 / b
        m |= dx * dx + dy * dy < 1
    return m


def ground_cover(g):
    E, lon, lat, n1, n2 = g.ELEV_CELL, g.CELL_LON, g.CELL_LAT, g.NOISE, g.NOISE2
    GREEN, DIRT, GRAVEL = g.SAND, g.THICK, g.ROUGH
    cover = np.full(E.shape, GREEN)
    p = np.full(E.shape, 0.02)
    near_water = g.water_cell.copy()
    for _ in range(2):
        near_water = near_water | g.shift(near_water, 1, 0, False) | g.shift(near_water, -1, 0, False) | \
            g.shift(near_water, 0, 1, False) | g.shift(near_water, 0, -1, False)
    hills = E > 120
    redwoods = hills & (((lon < -122.15) & (lat < 37.56)) | ((lat > 37.85) & (lon < -122.45)) |
                        ((lon > -122.25) & (lon < -122.12) & (lat > 37.74) & (lat < 37.93) & (E > 200)))
    cover[(E <= 120) & (n2 > 0.62)] = DIRT                       # fields and vacant lots
    cover[hills & (n1 > 0.58)] = DIRT                            # golden summer grass
    p[hills] = 0.06
    p[redwoods] = 0.2 + 0.3 * (1 - n1[redwoods])
    cover[(E > 600) & (n1 > 0.45)] = GRAVEL                      # upper Mount Diablo
    p[E > 600] = 0.05
    cover[near_water & (E < 30) & (lon < -122.44)] = GRAVEL       # Ocean Beach
    cover[near_water & (E < 4) & (lat < 37.6) & (lon > -122.2)] = GRAVEL   # salt ponds
    p[near_water] = np.minimum(p[near_water], 0.01)
    parks = _parks(g)
    cover[parks] = GREEN
    p[parks] = 0.3
    return cover, redwoods | (E > 300), p


def city_cover(g, cover, city):
    """Brick plazas downtown; parks stay green and keep their trees."""
    for lon, lat, r in DOWNTOWNS:
        cover[(g.dist_to_point(lon, lat) < r) & (g.NOISE > 0.55)] = g.PAVE
    parks = _parks(g)
    cover[parks] = g.SAND
    city[parks] = False


LANDMARKS = [('CACHIG05', 'Salesforce Tower (Sears Tower model)', -122.397, 37.790, 4, 4),
             ('CAFARM06', 'Point Bonita Lighthouse', -122.512, 37.830, 2, 2)]
CIVILIAN_START_CLEAR = 10
PROP_DENSITY = 0.35
CIVILIAN_POOLS = {
    'highrise': ['CANEWY01', 'CANEWY06', 'CANEWY07', 'CANEWY08', 'CANEWY13', 'CANEWY14', 'CANEWY15', 'CANWY05',
                 'CANWY22', 'CANWY24', 'CANWY25', 'CACHIG01', 'CACHIG02', 'CACHIG03', 'CAMIAM06', 'CAMIAM07',
                 'CAWASH04', 'CAWASH06'],
    'houses': ['CAHSE01', 'CAHSE02', 'CAHSE03', 'CAHSE04', 'CAHSE05', 'CAHSE06', 'CAHSE07', 'CAEUR1', 'CAEUR2'],
    'suburb': ['CAHSE01', 'CAHSE02', 'CAHSE04', 'CAGAS01', 'CAMSC10', 'CAMIAM01', 'CAMIAM02', 'CANEWY16',
               'CANEWY17', 'CANEWY18', 'CASWST01'],
    'campus': ['CATECH01', 'CAWASH03', 'CAWASH07', 'CAWASH09', 'CANWY23', 'CANEWY10', 'CANEWY11', 'CAMIAM05',
               'CACHIG04'],
    'industrial': ['CAIND01', 'CATS01', 'CACHIG06', 'CANWY09', 'CANEWY12', 'CABUNK01'],
    'farm': ['CAFARM01', 'CABARN02', 'CAFRMA', 'CAFARM02'],
}
CIVILIAN_ZONES = [
    (-122.4, 37.79, 6.6, 'highrise', 0.9), (-122.46, 37.755, 11.8, 'houses', 0.55),
    (-122.27, 37.805, 5.2, 'highrise', 0.85), (-122.31, 37.8, 3.9, 'industrial', 0.8),
    (-122.25, 37.845, 9.2, 'houses', 0.5), (-121.89, 37.335, 5.2, 'highrise', 0.8),
    (-121.88, 37.305, 11.8, 'houses', 0.45), (-122.05, 37.39, 7.9, 'campus', 0.55),
    (-121.96, 37.36, 5.2, 'campus', 0.45), (-122.15, 37.44, 5.2, 'houses', 0.4),
    (-122.36, 37.93, 5.2, 'industrial', 0.6), (-122.38, 37.73, 3.9, 'industrial', 0.6),
    (-122.32, 37.56, 5.2, 'suburb', 0.5), (-122.08, 37.665, 5.2, 'suburb', 0.5),
    (-122.03, 37.98, 5.2, 'suburb', 0.5), (-122.47, 37.69, 3.9, 'houses', 0.5),
    (-122.53, 37.97, 3.9, 'houses', 0.5), (-122.25, 38.085, 2.6, 'houses', 0.5),
    (-121.8, 37.655, 7.9, 'farm', 0.15), (-122.42, 37.45, 5.2, 'farm', 0.2),
]


def urban_mask(g):
    """The built-up bay plain: low ground except salt ponds, the Half Moon Bay coast, Marin and the far east."""
    E, lon, lat = g.ELEV_CELL, g.CELL_LON, g.CELL_LAT
    near_water = g.water_cell.copy()
    for _ in range(2):
        near_water = near_water | g.shift(near_water, 1, 0, False) | g.shift(near_water, -1, 0, False) | \
            g.shift(near_water, 0, 1, False) | g.shift(near_water, 0, -1, False)
    rural = ((lon < -122.38) & (lat < 37.56)) | ((lat > 37.85) & (lon < -122.44)) | \
        ((lon > -121.83) & (g.dist_to_point(-121.77, 37.68) > 9)) | \
        (near_water & (E < 4) & (lat < 37.6) & (lon > -122.2))
    return (E < 130) & ~rural


CIVILIAN_POOLS['fill'] = CIVILIAN_POOLS['houses'] + CIVILIAN_POOLS['suburb']
CIVILIAN_FILL = ('fill', 0.35)
CHECK_POINTS = [('San Rafael', -122.53, 37.98), ('Mt Tam', -122.59, 37.92), ('Vallejo', -122.25, 38.09),
                ('Angel Island', -122.428, 37.857), ('Alcatraz', -122.423, 37.827), ('Benicia', -122.16, 38.07)]
