"""Battle of the Bulge (突出部战役 / Ardennenoffensive, 16 Dec 1944 - 25 Jan 1945): the snowbound Ardennes from the
Elsenborn Ridge and the Schnee Eifel to the Meuse bridges at Dinant, Namur and Huy. The forest is dense, so the road
net decides the battle: Bastogne and St. Vith are the junctions every army needs. The Meuse, the German objective, is
real water with bridges at the 1944 crossings. Positions, rivers and places from OpenStreetMap."""
import numpy as np

from regions import osmwater

NAME = 'Battle of the Bulge'
OUTFILE = 'ardennes.mpr'
SEED = 1944  # Wacht am Rhein, 16 Dec 1944
W, H = 196, 151
LX, LY, LW, LH = 2, 4, 192, 138
LON0, LON1, LAT0, LAT1 = 4.60, 6.55, 50.62, 49.716
DEM_DIR, DEM_ZOOM = 'data/dem_ardennes', 11
BLUR_SMOOTH, BLUR_WIDE, BLUR_TERRAIN = 2, 10, 3
ELEV_PTS = [0, 130, 200, 270, 340, 410, 480, 550, 620, 700]   # Meuse 70-100 m, Famenne 200, plateau 450-550, Fagnes 690
LVL_PTS = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
THEATER = 'SNOW'   # the coldest Ardennes winter in memory; the skies only cleared on 23 Dec
LIGHTING = dict(Red=0.94, Green=0.97, Blue=1.06, Level=0.02, Ambient=0.96)

STARTS = [  # 北肩 / 圣维特 / 德军两路 / 南肩 / 巴斯托涅 / 西进 / 默兹河
    ('Elsenborn 埃尔森博恩岭', 6.215, 50.458),     # 美第2/99师守住北肩, 第6装甲集团军被挡住
    ('Sankt Vith 圣维特', 6.126, 50.280),          # 美第7装甲师守路口到12月21日
    ('Prüm 普吕姆', 6.425, 50.207),                # 第5装甲集团军出发地 (施内艾费尔)
    ('Clervaux 克莱沃', 6.028, 50.055),            # 第2装甲师渡过乌尔河, 天际线公路
    ('Ettelbruck 埃特尔布吕克', 6.099, 49.847),    # 南肩, 巴顿第3集团军北上解围
    ('Bastogne 巴斯托涅', 5.717, 50.002),          # 第101空降师被围, "Nuts!"
    ('Marche 马尔什', 5.342, 50.227),              # 西进路口, 美第84师
    ('Dinant 迪南 (默兹河西岸)', 4.780, 50.268),   # 英第29装甲旅守默兹河桥
]
START_FLAT, START_ORE_DIST, START_WATER_CLEAR = 9, 7.5, 3
MAP_CENTER = (5.75, 50.17)
ORE_TOWNS = [('Houffalize', 5.789, 50.132), ('La Roche-en-Ardenne', 5.576, 50.182), ('Wiltz', 5.932, 49.967),
             ('Vianden', 6.207, 49.934), ('Echternach', 6.400, 49.818), ('Trois-Ponts', 5.870, 50.373),
             ('Stoumont', 5.808, 50.407), ('Büllingen', 6.258, 50.408), ('Malmedy', 6.028, 50.427),
             ('Hotton', 5.446, 50.269), ('Manhay', 5.676, 50.293), ('Neufchâteau', 5.436, 49.840),
             ('Martelange', 5.738, 49.831), ('Losheim', 6.374, 50.361), ('Ciney', 5.101, 50.297),
             ('Beauraing', 4.955, 50.110), ('Durbuy', 5.456, 50.352), ('Sibret', 5.635, 49.968),
             ('Hosingen', 6.091, 50.014), ('Bleialf', 6.287, 50.239), ('Neuerburg', 6.295, 50.011),
             ('Waxweiler', 6.366, 50.092), ('Saint-Hubert', 5.374, 50.025), ('Tenneville', 5.524, 50.091),
             ('Wellin', 5.115, 50.082), ('Werbomont', 5.687, 50.380), ('Aywaille', 5.676, 50.476),
             ('Hamoir', 5.533, 50.425), ('Givet', 4.825, 50.137), ('Rosée', 4.688, 50.232), ('Mettet', 4.658, 50.321),
             ('Bioul', 4.797, 50.334), ('Gesves', 5.078, 50.403), ('Ohey', 5.124, 50.436), ('Wanze', 5.214, 50.555),
             ('Eghezée', 4.907, 50.592), ('Libramont', 5.376, 49.916), ('Bertogne', 5.668, 50.084),
             ('Lierneux', 5.792, 50.286), ('Heiderscheid', 5.977, 49.887), ('Burg-Reuland', 6.134, 50.197),
             ('Daleiden', 6.183, 50.070), ('Mettendorf', 6.333, 49.945), ('Libin', 5.257, 49.981)]
GEM_TOWNS = [('Vielsalm 磨刀石矿', 5.916, 50.284), ('Bertrix 板岩矿', 5.253, 49.855),
             ('Sprimont 采石场', 5.662, 50.505), ('Rochefort 大理石', 5.223, 50.159)]
OIL = [('Stavelot 燃料库', 5.931, 50.394), ('Francorchamps 燃料库', 5.953, 50.453),   # 派佩尔战斗群没能拿到的油料
       ('Gouvy 补给站', 5.943, 50.186)]
TECH = [('CAAIRP', 'Florennes A-78 机场', 4.670, 50.245, 3, 3),
        ('CAHOSP', 'Spa 美第1集团军司令部', 5.863, 50.492, 6, 4),
        ('CAOUTP', 'Baraque de Fraiture (Parker 十字路口)', 5.735, 50.255, 4, 3),
        ('CAOUTP', 'Foy-Notre-Dame (德军最远推进点)', 4.990, 50.247, 4, 3)]
BRIDGES = [  # Meuse crossings of 1944 (OSM positions)
    ('Monthermé', 4.731, 49.884),
    ('Givet', 4.826, 50.138),
    ('Dinant', 4.912, 50.262),
    ('Yvoir', 4.880, 50.327),
    ('Namur (Jambes)', 4.869, 50.462),
    ('Andenne', 5.096, 50.494),
    ('Huy', 5.241, 50.521),
]
CITIES = [('Namur', 4.866, 50.467, 2.6), ('Huy', 5.241, 50.518, 1.6), ('Bastogne', 5.717, 50.002, 1.6),
          ('Sankt Vith', 6.126, 50.280, 1.3), ('Malmedy', 6.028, 50.427, 1.3), ('Marche', 5.342, 50.227, 1.3),
          ('Dinant', 4.913, 50.259, 1.0), ('Spa', 5.863, 50.492, 1.2), ('Ettelbruck', 6.099, 49.847, 1.1),
          ('Diekirch', 6.160, 49.869, 1.1), ('Prüm', 6.425, 50.207, 1.0), ('Andenne', 5.097, 50.489, 1.1),
          ('Ciney', 5.101, 50.297, 1.1), ('Bitburg', 6.525, 49.973, 1.2), ('Wiltz', 5.932, 49.967, 0.9),
          ('Stavelot', 5.931, 50.394, 0.9), ('Givet', 4.825, 50.137, 0.9), ('Echternach', 6.421, 49.812, 0.9)]
P = {  # road nodes (OSM place positions)
    'Bastogne': (5.717, 50.002), 'Foy': (5.751, 50.045), 'Noville': (5.762, 50.064), 'Houffalize': (5.789, 50.132),
    'Fraiture': (5.735, 50.255), 'Manhay': (5.676, 50.293), 'Werbomont': (5.687, 50.380), 'Aywaille': (5.676, 50.476),
    'Longvilly': (5.837, 50.025), 'Clervaux': (6.028, 50.055), 'Marnach': (6.062, 50.054), 'Dasburg': (6.133, 50.048),
    'Arzfeld': (6.268, 50.087), 'Prüm': (6.425, 50.207), 'Wiltz': (5.932, 49.967), 'Heiderscheid': (5.977, 49.887),
    'Ettelbruck': (6.099, 49.847), 'Diekirch': (6.160, 49.869), 'Vianden': (6.207, 49.934), 'Bitburg': (6.525, 49.973),
    'Echternach': (6.421, 49.812), 'Hosingen': (6.091, 50.014), 'Assenois': (5.682, 49.967), 'Martelange': (5.738, 49.831),
    'Sibret': (5.635, 49.968), 'Neufchâteau': (5.436, 49.840), 'Champlon': (5.503, 50.108), 'Marche': (5.342, 50.227),
    'Hargimont': (5.309, 50.183), 'Buissonville': (5.196, 50.217), 'Celles': (5.006, 50.231), 'Foy-ND': (4.990, 50.247),
    'Dinant': (4.913, 50.259), 'Onhaye': (4.841, 50.244), 'Weillen': (4.780, 50.268), 'Ciney': (5.101, 50.297), 'Namur': (4.866, 50.467),
    'Rochefort': (5.223, 50.159), 'Beauraing': (4.955, 50.110), 'Givet': (4.825, 50.137), 'Hotton': (5.446, 50.269),
    'Soy': (5.510, 50.286), 'Erezée': (5.560, 50.293), 'La Roche': (5.576, 50.182), 'St. Vith': (6.126, 50.280),
    'Schönberg': (6.264, 50.289), 'Bleialf': (6.287, 50.239), 'Vielsalm': (5.916, 50.284), 'Poteau': (6.040, 50.320),
    'Recht': (6.043, 50.335), 'Ligneuville': (6.055, 50.376), 'Baugnez': (6.067, 50.403), 'Malmedy': (6.028, 50.427),
    'Stavelot': (5.931, 50.394), 'Trois-Ponts': (5.870, 50.373), 'La Gleize': (5.847, 50.411), 'Stoumont': (5.808, 50.407),
    'Francorchamps': (5.953, 50.453), 'Spa': (5.863, 50.492), 'Bütgenbach': (6.205, 50.427), 'Elsenborn': (6.221, 50.457),
    'Büllingen': (6.258, 50.408), 'Honsfeld': (6.283, 50.382), 'Losheim': (6.374, 50.361), 'Amel': (6.171, 50.354),
    'Gouvy': (5.943, 50.186), 'Burg-Reuland': (6.134, 50.197), 'Saint-Hubert': (5.374, 50.025), 'Libramont': (5.376, 49.916),
    'Andenne': (5.097, 50.489), 'Huy': (5.241, 50.518), 'Durbuy': (5.456, 50.352), 'Hamoir': (5.533, 50.425),
    'Monschau': (6.241, 50.554), 'Gesves': (5.078, 50.403), 'Yvoir': (4.878, 50.326), 'Mettet': (4.658, 50.321),
    'Florennes': (4.603, 50.251),
}


def _road(*names):
    return [P[n] for n in names]


TRACKS = {  # the main roads of December 1944
    'N30 Bastogne-Liège': _road('Bastogne', 'Foy', 'Noville', 'Houffalize', 'Fraiture', 'Manhay', 'Werbomont', 'Aywaille'),
    'N4 Arlon-Bastogne-Marche-Namur': [(5.775, 49.716)] + _road('Martelange', 'Assenois', 'Bastogne', 'Champlon', 'Marche',
                                                               'Ciney', 'Namur'),
    'Bastogne-Clervaux-Dasburg-Prüm': _road('Bastogne', 'Longvilly', 'Clervaux', 'Marnach', 'Dasburg', 'Arzfeld', 'Prüm'),
    'Bastogne-Wiltz-Ettelbruck': _road('Bastogne', 'Wiltz', 'Heiderscheid', 'Ettelbruck'),
    'Bastogne-Neufchâteau': _road('Bastogne', 'Sibret', 'Neufchâteau'),
    'Marche-Celles-Dinant': _road('Marche', 'Hargimont', 'Buissonville', 'Celles', 'Foy-ND', 'Dinant', 'Weillen', 'Florennes'),
    'Dinant-Ciney': _road('Dinant', 'Ciney'),
    'Meuse west bank': _road('Givet', 'Onhaye', 'Yvoir', 'Namur'),
    'Givet-Beauraing-Rochefort-Marche': _road('Givet', 'Beauraing', 'Rochefort', 'Hargimont'),
    'Marche-Hotton-Manhay': _road('Marche', 'Hotton', 'Soy', 'Erezée', 'Manhay'),
    'La Roche-Hotton-Durbuy': _road('Champlon', 'La Roche', 'Hotton', 'Durbuy', 'Hamoir'),
    'Houffalize-La Roche': _road('Houffalize', 'La Roche'),
    'St. Vith-Schönberg-Prüm': _road('St. Vith', 'Schönberg', 'Bleialf', 'Prüm'),
    'St. Vith-Vielsalm-Manhay': _road('St. Vith', 'Vielsalm', 'Fraiture'),
    'St. Vith-Poteau-Malmedy': _road('St. Vith', 'Poteau', 'Recht', 'Ligneuville', 'Baugnez', 'Malmedy'),
    'St. Vith-Gouvy-Houffalize': _road('St. Vith', 'Gouvy', 'Houffalize'),
    'St. Vith-Burg-Reuland-Clervaux': _road('St. Vith', 'Burg-Reuland', 'Marnach'),
    'St. Vith-Amel-Büllingen': _road('St. Vith', 'Amel', 'Büllingen', 'Elsenborn', 'Monschau'),
    'Peiper route': _road('Losheim', 'Honsfeld', 'Büllingen', 'Ligneuville', 'Stavelot', 'Trois-Ponts', 'La Gleize',
                          'Stoumont', 'Aywaille'),
    'Malmedy-Bütgenbach-Elsenborn': _road('Malmedy', 'Bütgenbach', 'Elsenborn'),
    'Malmedy-Francorchamps-Spa': _road('Malmedy', 'Francorchamps', 'Spa'),
    'Trois-Ponts-Vielsalm': _road('Trois-Ponts', 'Vielsalm'),
    'Trois-Ponts-Werbomont': _road('Trois-Ponts', 'Werbomont'),
    'Skyline Drive': _road('Clervaux', 'Hosingen', 'Diekirch', 'Ettelbruck'),
    'Diekirch-Vianden-Bitburg': _road('Diekirch', 'Vianden', 'Bitburg'),
    'Echternach-Bitburg': _road('Echternach', 'Bitburg'),
    'Ettelbruck-Echternach': _road('Diekirch', 'Echternach'),
    'Bastogne-Saint-Hubert': _road('Bastogne', 'Champlon'),
    'Saint-Hubert-Libramont-Neufchâteau': _road('Champlon', 'Saint-Hubert', 'Libramont', 'Neufchâteau'),
    'Saint-Hubert-Rochefort': _road('Saint-Hubert', 'Rochefort'),
    'Namur-Andenne-Huy': _road('Namur', 'Andenne', 'Huy'),
    'Ciney-Huy': _road('Ciney', 'Gesves', 'Andenne'),
    'Huy-Durbuy': _road('Huy', 'Durbuy'),
    'Namur-Mettet': _road('Namur', 'Mettet', 'Florennes'),
}
TRACK_COVER = 'PAVE'
ROADS = TRACKS    # no trees on the roads
_W = 'data/osm_ardennes/water.json'
RIVER_NAMES = set()
RIVER_BBOX = (4.4, 49.5, 6.8, 50.8)
EXTRA_RIVERS = osmwater.lines(_W, {"L'Ourthe", "L'Ourthe orientale", "L'Ourthe occidentale", 'Amblève', 'Our', 'Sûre',
                                  'La Sûre', 'Lesse', 'La Lesse', 'La Lomme', 'Wiltz', 'Clerve', 'la Semois', 'Warche',
                                  'Amel', 'Prüm', 'Salm', 'La Salm', "L'Aisne", 'Alzette', 'Vesdre', 'Hoëgne', 'Le Bocq',
                                  'La Molignée', 'Le Hoyoux', 'La Houille', 'Hermeton'})
WATER_RIVERS = osmwater.lines(_W, {'La Meuse'})
WATER_HALF_WIDTH = 0.9
LAKES = []
WATER_LEVEL = 0
MIN_WATER_CORNERS, MIN_LAND_CORNERS = 6, 8
MOUNTAIN_TREES = ['TREE25', 'TREE26', 'TREE27', 'TREE05', 'TREE06', 'TREE07', 'TREE08', 'TREE15']   # spruce
VALLEY_TREES = ['TREE01', 'TREE02', 'TREE03', 'TREE04', 'TREE10', 'TREE11', 'TREE12', 'TREE17']    # oak and beech
CHECK_POINTS = [(n, lon, lat) for n, lon, lat in OIL] + [(n, lon, lat) for n, lon, lat in GEM_TOWNS]


def ground_cover(g):
    E, R, lon, lat, n1, n2 = g.ELEV_CELL, g.RELIEF_CELL, g.CELL_LON, g.CELL_LAT, g.NOISE, g.NOISE2
    cover = np.full(E.shape, g.GRASS)
    p = np.full(E.shape, 0.01)
    slopes = R > 22                                                 # the deep-cut valley sides
    upland = E > 330                                                # Ardennes / Eifel plateau
    fagnes = (E > 560) & (lat > 50.42) & (lon > 5.95)              # Hautes Fagnes: open snowy moor
    famenne = (E < 280) & (lat > 50.05) & (lat < 50.30) & (lon > 5.0) & (lon < 5.55)   # Famenne: open farm country
    forest = (slopes | (upland & (n1 < 0.5)) | (n2 < 0.22)) & ~fagnes & ~famenne
    riparian = (g.RIVER_D < 1.3) & ~forest

    cover[~forest & (n1 > 0.6)] = g.THICK                          # stubble fields through thin snow
    cover[fagnes & (n2 > 0.4)] = g.ROUGH                           # windswept heath
    cover[slopes & (n2 > 0.7)] = g.ROUGH                           # rock in the gorges
    p[~forest] = 0.012
    p[riparian] = 0.06
    p[fagnes] = 0.03
    p[forest] = 0.40 + 0.14 * (1 - n2[forest])                     # dense spruce: off the roads, tanks crawl
    return cover, forest, p


# --- buildings ---------------------------------------------------------------------------
from regions.pools import POOLS as CIVILIAN_POOLS   # noqa: E402

CIVILIAN_START_CLEAR = 7
FLAT_CITIES = []
PROP_DENSITY = 0.3
CIVILIAN_ZONES = [  # lon, lat, radius (cells), pool, density
    (4.866, 50.467, 2.5, 'suburb', 0.6),         # Namur
    (5.241, 50.518, 2, 'industrial', 0.6),       # Huy
    (6.028, 50.427, 1.8, 'village', 0.6),        # Malmedy
    (5.863, 50.492, 1.6, 'village', 0.5),        # Spa
    (6.160, 49.869, 1.6, 'village', 0.5),        # Diekirch
    (5.101, 50.297, 1.6, 'village', 0.5),        # Ciney
    (5.097, 50.489, 1.6, 'industrial', 0.5),     # Andenne
    (5.55, 50.25, 5, 'farm', 0.25),
    (5.20, 50.10, 5, 'farm', 0.25),
    (5.90, 50.15, 5, 'farm', 0.2),
    (6.25, 50.15, 5, 'farm', 0.2),
    (5.55, 49.90, 5, 'village', 0.25),
    (6.00, 50.35, 4, 'village', 0.25),
    (5.00, 50.40, 5, 'farm', 0.25),
]
CIVILIAN_FILL = ('small', 0.6)
