"""San Francisco Bay Area: 8 players around the bay, Pacific + bay water from bathymetry, Marin cut off by water."""
import numpy as np

NAME = 'San Francisco Bay'
OUTFILE = 'bayarea.mpr'
SEED = 1849  # gold rush
W, H = 76, 183
LX, LY, LW, LH = 2, 4, 72, 170
LON0, LON1, LAT0, LAT1 = -122.66, -121.70, 38.10, 37.20
DEM_DIR, DEM_ZOOM = 'data/dem_bayarea', 10
BLUR_SMOOTH, BLUR_WIDE, BLUR_TERRAIN = 2, 8, 4
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
START_FLAT, START_ORE_DIST, START_WATER_CLEAR = 10, 7.5, 3
START_FIELDS_MIN = 1   # San Francisco has room for one; Golden Gate Park makes up the second
MAP_CENTER = (-122.15, 37.65)
ORE_TOWNS = [('Golden Gate Park', -122.48, 37.77), ('Daly City', -122.47, 37.69), ('San Mateo', -122.32, 37.56), ('Redwood City', -122.24, 37.49),
             ('Mountain View', -122.08, 37.39), ('Milpitas', -121.90, 37.43), ('Dublin', -121.93, 37.70),
             ('Berkeley', -122.27, 37.87), ('Concord', -122.03, 37.98), ('San Leandro', -122.15, 37.72),
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
ISLANDS = [(-122.423, 37.827, 3.6), (-122.428, 37.857, 2.6)]   # Alcatraz, Angel Island (too small for the DEM at this scale)
MIN_WATER_CORNERS, MIN_LAND_CORNERS = 6, 6
CITIES = [('San Francisco', -122.43, 37.765, 7), ('Oakland', -122.25, 37.80, 5.5), ('Berkeley', -122.27, 37.87, 3),
          ('San Jose', -121.89, 37.33, 6.5), ('Fremont', -121.96, 37.55, 3.5), ('Palo Alto', -122.15, 37.43, 3),
          ('San Mateo', -122.32, 37.56, 3), ('Walnut Creek', -122.05, 37.91, 3), ('Livermore', -121.77, 37.68, 3),
          ('Hayward', -122.08, 37.665, 3), ('Richmond', -122.35, 37.94, 3), ('Daly City', -122.47, 37.69, 2.5),
          ('Sunnyvale', -122.03, 37.38, 4), ('Santa Clara', -121.96, 37.35, 3), ('Concord', -122.03, 37.98, 3),
          ('San Rafael', -122.53, 37.97, 2.5), ('Vallejo', -122.25, 38.10, 2.5), ('Half Moon Bay', -122.42, 37.47, 2)]
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


def ground_cover(g):
    E, lon, lat, n1, n2 = g.ELEV_CELL, g.CELL_LON, g.CELL_LAT, g.NOISE, g.NOISE2
    cover = np.full(E.shape, g.GRASS)
    p = np.full(E.shape, 0.02)
    near_water = g.water_cell.copy()
    for _ in range(2):
        near_water = near_water | g.shift(near_water, 1, 0, False) | g.shift(near_water, -1, 0, False) | \
            g.shift(near_water, 0, 1, False) | g.shift(near_water, 0, -1, False)
    hills = E > 120
    redwoods = hills & (((lon < -122.15) & (lat < 37.56)) | ((lat > 37.85) & (lon < -122.45)) |
                        ((lon > -122.25) & (lon < -122.12) & (lat > 37.74) & (lat < 37.93) & (E > 200)))

    cover[(E <= 120) & (n1 > 0.66)] = g.THICK                   # parks and orchards
    cover[hills & (n1 > 0.4)] = g.ROUGH                          # golden summer grass
    p[hills] = 0.06
    cover[redwoods & (n1 < 0.55)] = g.THICK                      # redwoods and oak woodland
    p[redwoods] = 0.2 + 0.3 * (1 - n1[redwoods])
    cover[E > 600] = g.ROUGH                                     # upper Mount Diablo
    p[E > 600] = 0.05
    ocean_beach = near_water & (E < 30) & (lon < -122.44)
    cover[ocean_beach] = g.SAND
    salt_ponds = near_water & (E < 4) & (lat < 37.6) & (lon > -122.2)
    cover[salt_ponds] = g.ROUGH
    p[near_water] = np.minimum(p[near_water], 0.01)
    return cover, redwoods | (E > 300), p

CHECK_POINTS = [('San Rafael', -122.53, 37.98), ('Mt Tam', -122.59, 37.92), ('Vallejo', -122.25, 38.09),
                ('Angel Island', -122.432, 37.862), ('Alcatraz', -122.423, 37.827), ('Benicia', -122.16, 38.07)]
