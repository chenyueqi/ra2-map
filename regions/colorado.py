"""Colorado: 8 players, real Rockies + plains, no open water."""
import numpy as np

NAME = 'Colorado'
OUTFILE = 'colorado.mpr'
SEED = 1876  # statehood
W, H = 140, 111
LX, LY, LW, LH = 2, 4, 136, 98
LON0, LON1, LAT0, LAT1 = -109.05, -102.04, 41.0, 37.0
DEM_DIR, DEM_ZOOM = 'data/dem', 8
BLUR_SMOOTH, BLUR_WIDE, BLUR_TERRAIN = 3, 8, 6
ELEV_PTS = [1000, 1700, 1950, 2250, 2550, 2850, 3150, 3450, 3750, 4100]
LVL_PTS = [0, 0, 1, 2, 3.2, 4.6, 6.0, 7.4, 8.8, 10.2]
LIGHTING = dict(Red=1.0, Green=0.98, Blue=0.95, Level=0.032, Ambient=1.05)

STARTS = [  # name, lon, lat (nudged onto open ground where the real town hugs a mountain or the border)
    ('Boulder', -105.13, 40.03),
    ('Colorado Springs', -104.72, 38.86),
    ('Sterling', -103.21, 40.58),
    ('Lamar', -102.80, 38.12),
    ('Grand Junction', -108.42, 39.10),
    ('Durango', -107.86, 37.33),
    ('Steamboat Springs', -106.88, 40.50),
    ('Alamosa', -105.87, 37.50),
]
START_FLAT, START_ORE_DIST = 12, 8.5
MAP_CENTER = (-105.55, 39.0)
ORE_TOWNS = [('Limon', -103.69, 39.26), ('Fort Morgan', -103.80, 40.25), ('La Junta', -103.54, 37.98),
             ('Walsenburg', -104.78, 37.62), ('Montrose', -107.88, 38.48), ('Meeker', -107.91, 40.04),
             ('Granby', -105.94, 40.09), ('Salida', -106.00, 38.53), ('Cortez', -108.59, 37.35),
             ('Trinidad', -104.50, 37.17), ('Burlington', -102.40, 39.30), ('Pueblo east', -104.45, 38.30),
             ('Gunnison', -106.93, 38.55), ('Craig', -107.55, 40.52), ('Fort Collins east', -104.93, 40.62),
             ('Kremmling', -106.39, 40.06), ('Glenwood Springs', -107.32, 39.55)]
GEM_TOWNS = [('Leadville', -106.29, 39.25), ('Cripple Creek', -105.18, 38.75), ('Silverton', -107.66, 37.81),
             ('Central City', -105.51, 39.80), ('Telluride', -107.81, 37.94)]
OIL = [('Greeley', -104.70, 40.42), ('Wattenberg', -104.85, 40.17), ('Rangely', -108.80, 40.09),
       ('Rifle', -107.78, 39.53), ('Trinidad field', -104.30, 37.30), ('Aneth', -108.85, 37.30),
       ('Akron', -103.21, 40.16), ('Kit Carson', -102.79, 38.76)]
TECH = [('CAAIRP', 'Denver International Airport', -104.67, 39.86, 3, 3),
        ('CAHOSP', 'Anschutz Medical Campus', -104.83, 39.745, 6, 4),
        ('CAOUTP', 'Pueblo Chemical Depot', -104.33, 38.32, 4, 3)]
CITIES = [('Denver', -104.97, 39.73, 6.5), ('Boulder', -105.13, 40.03, 3.5), ('Colorado Springs', -104.72, 38.86, 4),
          ('Fort Collins', -105.08, 40.58, 2.5), ('Pueblo', -104.61, 38.27, 2.5), ('Greeley', -104.70, 40.42, 2),
          ('Grand Junction', -108.42, 39.10, 3), ('Durango', -107.86, 37.33, 2.5), ('Alamosa', -105.87, 37.50, 2.5),
          ('Sterling', -103.21, 40.58, 2.5), ('Lamar', -102.80, 38.12, 2.5), ('Steamboat Springs', -106.88, 40.50, 2.5)]
ROADS = {  # tree-free corridors along the real highways
    'I-70': [(-104.99, 39.74), (-105.51, 39.74), (-106.07, 39.63), (-106.37, 39.64), (-107.32, 39.55),
             (-107.78, 39.53), (-108.42, 39.10), (-109.2, 39.1)],
    'US-50': [(-102.80, 38.12), (-103.54, 37.98), (-104.61, 38.27), (-105.24, 38.44), (-106.0, 38.53),
              (-106.33, 38.50), (-106.93, 38.55), (-107.88, 38.48), (-108.42, 39.10)],
    'US-160': [(-104.78, 37.62), (-105.2, 37.60), (-105.87, 37.50), (-106.8, 37.48), (-107.01, 37.27),
               (-107.86, 37.33), (-108.59, 37.35)],
    'US-40': [(-104.99, 39.74), (-105.78, 39.80), (-105.94, 40.09), (-106.39, 40.06), (-106.6, 40.38),
              (-106.88, 40.50), (-107.55, 40.52), (-108.80, 40.09)],
    'US-285': [(-104.99, 39.74), (-105.5, 39.45), (-106.0, 39.22), (-106.08, 38.50), (-106.14, 38.09),
               (-105.87, 37.50)],
    'US-550': [(-107.88, 38.48), (-107.67, 38.02), (-107.66, 37.81), (-107.86, 37.33)],
    'I-25': [(-105.08, 40.58), (-104.99, 39.74), (-104.72, 38.86), (-104.61, 38.27), (-104.78, 37.62),
             (-104.50, 37.17)],
    'I-76': [(-104.99, 39.74), (-103.80, 40.25), (-103.21, 40.58)],
}
RIVER_NAMES = {'Arkansas', 'Colorado', 'Rio Grande', 'South Platte', 'San Juan', 'North Platte', 'Green',
               'Animas', 'Dolores', 'Gunnison', 'White', 'Yampa', 'Purgatoire', 'Republican',
               'S. Fork Republican', 'Arikaree', 'Big Sandy', 'Little Snake', 'Laramie', 'Cimarron', 'Taylor'}
RIVER_BBOX = (-110, 36.5, -101, 41.5)
EXTRA_RIVERS = []
WATER_RIVERS, WATER_HALF_WIDTH, LAKES = [], 0, []
MOUNTAIN_TREES = ['TREE25', 'TREE26', 'TREE27', 'TREE05', 'TREE06', 'TREE07', 'TREE08', 'TREE15']
VALLEY_TREES = ['TREE01', 'TREE02', 'TREE03', 'TREE04', 'TREE10', 'TREE11', 'TREE12', 'TREE17']


def ground_cover(g):
    """Returns (cover, forest, tree_probability) arrays from the engine context g."""
    E, R, lon, lat, n1, n2 = g.ELEV_CELL, g.RELIEF_CELL, g.CELL_LON, g.CELL_LAT, g.NOISE, g.NOISE2
    cover = np.full(E.shape, g.GRASS)
    forest = (E > 2350) & (E < 3500) & ((R > 25) | (E > 2650))
    riparian = g.RIVER_D < 1.3
    cover[forest & (n1 > 0.3)] = g.THICK
    cover[riparian & (E < 2400)] = g.THICK
    cover[E >= 3500] = g.ROUGH
    east = lon > -104.6
    cover[east & ~riparian & (n1 > 0.58)] = g.SAND
    cover[east & ~riparian & (n2 > 0.72) & (n1 <= 0.58)] = g.ROUGH
    west_desert = (lon < -107.6) & (E < 2000) & ~riparian
    cover[west_desert & (n1 > 0.42)] = g.SAND
    cover[west_desert & (n1 <= 0.42) & (n2 > 0.55)] = g.ROUGH
    slv = (lat > 37.1) & (lat < 38.3) & (lon > -106.3) & (lon < -105.5) & (E < 2500)
    cover[slv & ~riparian & (n1 > 0.5)] = g.SAND
    cover[g.dist_to_point(-105.55, 37.74) < 3.2 + 1.5 * n2] = g.SAND        # Great Sand Dunes
    p = np.zeros(E.shape)
    p[forest] = 0.12 + 0.45 * n1[forest]
    p[riparian] = np.maximum(p[riparian], 0.28)
    p[E >= 3500] = 0.03
    p[p == 0] = 0.006
    mountain = forest | (E > 2300)
    return cover, mountain, p

# --- buildings ---------------------------------------------------------------------------
from regions.pools import POOLS as CIVILIAN_POOLS   # noqa: E402

LANDMARKS = [('CACOLO01', 'Air Force Academy Chapel', -104.89, 39.008, 5, 3),
             ('CATECH01', 'NORAD Cheyenne Mountain', -104.85, 38.744, 3, 3)]
CIVILIAN_START_CLEAR = 11
FLAT_CITIES = [('Denver', -104.97, 39.73, 7), ('Fort Collins', -105.08, 40.58, 3), ('Pueblo', -104.61, 38.27, 3),
               ('Greeley', -104.70, 40.42, 2.5)]
PROP_DENSITY = 0.3
CIVILIAN_ZONES = [  # lon, lat, radius (cells), pool, density
    (-104.99, 39.745, 3, 'highrise', 0.9),      # downtown Denver
    (-104.95, 39.70, 6, 'small', 0.8),           # Denver neighbourhoods
    (-104.85, 39.80, 4, 'industrial', 0.5),      # Commerce City
    (-104.61, 38.26, 3, 'industrial', 0.7),      # Pueblo steel mill
    (-105.08, 40.58, 3, 'small', 0.8),           # Fort Collins
    (-104.80, 38.84, 4, 'suburb', 0.4),          # Colorado Springs
    (-106.82, 39.19, 2.5, 'small', 0.9),         # Aspen
    (-106.37, 39.64, 2.5, 'small', 0.9),         # Vail
    (-106.04, 39.48, 2.5, 'small', 0.9),         # Breckenridge
    (-108.07, 38.74, 3, 'farm', 0.3),            # Delta
    (-104.70, 40.42, 4, 'farm', 0.3),            # Greeley
    (-103.50, 40.40, 6, 'farm', 0.2),            # eastern plains
    (-102.60, 39.70, 6, 'farm', 0.2),
    (-103.00, 38.90, 6, 'farm', 0.2),
    (-103.90, 38.60, 5, 'farm', 0.2),
    (-104.30, 39.60, 5, 'farm', 0.2),
    (-105.90, 37.75, 5, 'farm', 0.2),            # San Luis Valley potato farms
]
CIVILIAN_FILL = ('small', 0.8)   # at ~3 km per cell only small houses fit a town
