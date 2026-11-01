"""Battle of Verdun (凡尔登战役, 21 Feb - 18 Dec 1916): the Meuse valley through Verdun, the Côtes de Meuse heights
with the ring of forts on the right bank, Mort-Homme and Côte 304 on the left bank, the German side at Montfaucon,
Damvillers, Étain (the Woëvre plain) and the Saint-Mihiel salient, and the Voie Sacrée from Bar-le-Duc.
Positions of towns, forts, the Meuse, the railway and the Voie Sacrée come from OpenStreetMap (data/osm_verdun)."""
import numpy as np

from regions import osmwater

NAME = 'Battle of Verdun'
OUTFILE = 'verdun.mpr'
SEED = 1916  # 21 Feb 1916, Operation Gericht
W, H = 164, 199
LX, LY, LW, LH = 2, 4, 160, 186
LON0, LON1, LAT0, LAT1 = 5.00, 5.76, 49.42, 48.8391
DEM_DIR, DEM_ZOOM = 'data/dem_verdun', 12
BLUR_SMOOTH, BLUR_WIDE, BLUR_TERRAIN = 3, 14, 5
ELEV_PTS = [0, 215, 245, 285, 325, 365, 400]   # Meuse valley and Woëvre plain at the water's level, the Côtes 4-5 up
LVL_PTS = [0, 0, 1, 2, 3, 4, 5]
LIGHTING = dict(Red=0.98, Green=0.97, Blue=0.97, Level=0.02, Ambient=0.98)   # grey, smoke over the shell fields

STARTS = [  # 法军 / 德军
    ('Verdun 凡尔登', 5.410, 49.162),                    # the citadel town (start on the right bank, east of the Meuse)
    ('Fort de Douaumont 杜奥蒙堡', 5.4386, 49.2169),       # taken 25 Feb, retaken 24 Oct 1916
    ('Cote 304 304高地', 5.2096, 49.2261),                # left bank, Mar-May 1916 (with Mort-Homme next to it)
    ('Montfaucon 蒙福孔', 5.1409, 49.2732),               # German observation post (the Kronprinz's butte)
    ('Damvillers 当维莱', 5.4005, 49.3427),               # German 5th Army jump-off behind the Bois des Caures
    ('Etain 埃坦', 5.6363, 49.2133),                      # the Woëvre plain
    ('Saint-Mihiel 圣米耶勒', 5.575, 48.897),             # tip of the German salient across the Meuse
    ('Souilly 苏伊利', 5.2858, 49.0277),                  # Pétain's headquarters on the Voie Sacrée
]
START_FLAT, START_ORE_DIST, START_WATER_CLEAR = 9, 7.5, 3
MAP_CENTER = (5.39, 49.14)
ORE_TOWNS = [('Clermont-en-Argonne', 5.0712, 49.1068), ('Varennes-en-Argonne', 5.0362, 49.2269),
             ('Avocourt', 5.1424, 49.2068), ('Dombasle-en-Argonne', 5.1911, 49.1381), ('Nixéville', 5.2754, 49.1098),
             ('Nubécourt', 5.1742, 48.9998), ('Chaumont-sur-Aire', 5.2562, 48.9278), ('Pierrefitte-sur-Aire', 5.3299, 48.9003),
             ('Senoncourt', 5.3327, 49.0581), ('Ancemont', 5.4079, 49.0704), ('Sommedieue', 5.4633, 49.0840),
             ('Troyon', 5.4643, 49.0034), ('Lacroix-sur-Meuse', 5.5115, 48.9726), ('Rupt-en-Woëvre', 5.4904, 49.0530),
             ('Haudiomont', 5.5603, 49.1157), ('Fresnes-en-Woëvre', 5.6307, 49.0963),
             ('Hannonville-sous-les-Côtes', 5.6583, 49.0387), ('Spincourt', 5.6670, 49.3289), ('Mangiennes', 5.5269, 49.3549),
             ('Azannes', 5.4669, 49.2916), ('Consenvoye', 5.2862, 49.2875), ('Dun-sur-Meuse', 5.1861, 49.3864),
             ('Romagne-sous-Montfaucon', 5.0832, 49.3318), ('Rupt-devant-Saint-Mihiel', 5.4043, 48.8825),
             ('Haudainville', 5.4200, 49.1265), ('Charny-sur-Meuse', 5.3618, 49.2071), ('Sivry-sur-Meuse', 5.2680, 49.3214),
             ('Récicourt', 5.1563, 49.1414), ('Génicourt-sur-Meuse', 5.4362, 49.0353)]
GEM_TOWNS = [('Mort-Homme', 5.2511, 49.2315),          # the key heights fought over for months
             ('Bois des Caures', 5.3926, 49.2665),     # Driant's chasseurs, first day of the battle
             ('Les Éparges', 5.5811, 49.0525),         # the mine-cratered ridge over the Woëvre
             ('Butte de Vauquois', 5.0767, 49.2012)]   # the Argonne mine war
OIL = []
TECH = [('CAAIRP', 'Vadelaincourt 机场', 5.240, 49.072, 3, 3),
        ('CAHOSP', 'Baleycourt 后送医院', 5.300, 49.122, 6, 4),
        ('CAOUTP', 'Côte du Poivre 观察所', 5.3816, 49.2250, 4, 3)]
LANDMARKS = [  # the ring of forts, as concrete bunkers
    ('CABUNK01', 'Fort de Vaux', 5.4702, 49.2005, 2, 2),
    ('CABUNK02', 'Fort de Souville', 5.4391, 49.1878, 2, 2),
    ('CABUNK01', 'Fort de Tavannes', 5.4631, 49.1814, 2, 2),
    ('CABUNK02', 'Fort de Moulainville', 5.4844, 49.1679, 2, 2),
    ('CABUNK01', 'Ouvrage de Thiaumont', 5.4197, 49.2067, 2, 2),
    ('CABUNK02', 'Ouvrage de Froideterre', 5.4039, 49.1976, 2, 2),
    ('CABUNK01', 'Fort de Marre', 5.3148, 49.1957, 2, 2),
    ('CABUNK02', 'Fort de Bois Bourrus', 5.2992, 49.1923, 2, 2),
    ('CABUNK01', 'Fort de Belrupt', 5.4349, 49.1479, 2, 2),
    ('CABUNK02', 'Fort de Génicourt', 5.4476, 49.0569, 2, 2),
]
BRIDGES = [  # Meuse crossings, north to south
    ('Dun-sur-Meuse', 5.183, 49.386),
    ('Consenvoye', 5.297, 49.288),
    ('Bras-sur-Meuse', 5.368, 49.209),
    ('Belleville', 5.378, 49.180),
    ('Verdun', 5.387, 49.160),
    ('Haudainville', 5.405, 49.125),
    ('Dieue-sur-Meuse', 5.420, 49.072),
    ('Troyon', 5.455, 49.003),
    ('Lacroix-sur-Meuse', 5.505, 48.972),
    ('Saint-Mihiel', 5.538, 48.893),
]
CITIES = [('Verdun', 5.388, 49.160, 2.5), ('Étain', 5.6363, 49.2133, 1.3), ('Saint-Mihiel', 5.545, 48.892, 1.3),
          ('Clermont-en-Argonne', 5.0712, 49.1068, 0.9), ('Dun-sur-Meuse', 5.1861, 49.3864, 0.9),
          ('Thierville', 5.348, 49.1706, 0.9), ('Belleville', 5.3703, 49.1814, 0.9)]
RAILWAYS = {  # Ligne de Saint-Hilaire-au-Temple à Hagondange: Paris - Sainte-Menehould - Verdun - Étain - Metz
    'Sainte-Menehould - Verdun - Étain': [
        (4.9641, 49.093), (5.0152, 49.1103), (5.0224, 49.1091), (5.0399, 49.0994), (5.0498, 49.0996), (5.0609, 49.1057),
        (5.069, 49.1077), (5.0767, 49.1145), (5.0747, 49.1249), (5.0779, 49.1274), (5.0854, 49.1285), (5.0876, 49.131),
        (5.078, 49.1426), (5.079, 49.1459), (5.0829, 49.1474), (5.0935, 49.1468), (5.1157, 49.14), (5.1363, 49.1446),
        (5.1743, 49.1409), (5.2062, 49.1302), (5.2195, 49.1167), (5.2293, 49.1167), (5.2398, 49.1131), (5.2546, 49.1169),
        (5.2638, 49.1122), (5.269, 49.1118), (5.2769, 49.1176), (5.2914, 49.1204), (5.2992, 49.1321), (5.3362, 49.148),
        (5.365, 49.1568), (5.3699, 49.1628), (5.3787, 49.1647), (5.3827, 49.1739), (5.3862, 49.1761), (5.3996, 49.1741),
        (5.4104, 49.1676), (5.421, 49.1674), (5.45, 49.1787), (5.4646, 49.1876), (5.4816, 49.1832), (5.4952, 49.1843),
        (5.5372, 49.198), (5.5463, 49.198), (5.5843, 49.207), (5.611, 49.2107), (5.6236, 49.2079), (5.6349, 49.2085),
        (5.6411, 49.207), (5.6521, 49.2002), (5.6652, 49.186), (5.6859, 49.1776), (5.7595, 49.162), (5.8051, 49.1647)],
}
TRACKS = {  # the Voie Sacrée, Bar-le-Duc - Souilly - Verdun: 3000 lorries a day kept Verdun alive
    'Voie Sacrée': [
        (5.2679, 48.8347), (5.2792, 48.8374), (5.2821, 48.8405), (5.2823, 48.8492), (5.2789, 48.8517), (5.2702, 48.8691),
        (5.2595, 48.8823), (5.2499, 48.8861), (5.2398, 48.8977), (5.2346, 48.9078), (5.2349, 48.9124), (5.253, 48.9269),
        (5.2558, 48.9317), (5.2618, 48.9342), (5.2834, 48.9507), (5.2881, 48.9589), (5.2851, 48.9678), (5.2893, 48.9754),
        (5.2955, 48.9928), (5.2931, 48.9994), (5.2961, 49.0067), (5.2838, 49.0239), (5.2927, 49.0398), (5.2882, 49.0434),
        (5.2878, 49.0587), (5.2839, 49.063), (5.2878, 49.0749), (5.285, 49.0921), (5.2936, 49.1031), (5.2916, 49.1151),
        (5.2963, 49.1272), (5.3150, 49.1400), (5.3450, 49.1500)],
}
TRACK_COVER = 'PAVE'
ROADS = dict(RAILWAYS, **TRACKS, **{  # roads of 1916 (no trees on them)
    'N3 Verdun - Étain': [(5.39, 49.16), (5.47, 49.175), (5.55, 49.19), (5.6363, 49.2133), (5.76, 49.23)],
    'N3 Verdun - Clermont': [(5.38, 49.16), (5.27, 49.14), (5.19, 49.138), (5.0712, 49.1068), (5.00, 49.10)],
    'Verdun - Saint-Mihiel': [(5.39, 49.155), (5.42, 49.1265), (5.4272, 49.0707), (5.4362, 49.0353), (5.4643, 49.0034),
                              (5.5115, 48.9726), (5.545, 48.892)],
    'Verdun - Damvillers': [(5.388, 49.165), (5.3756, 49.2096), (5.40, 49.236), (5.4005, 49.3427)],
    'Verdun - Montfaucon': [(5.37, 49.16), (5.30, 49.19), (5.2128, 49.2084), (5.1409, 49.2732)],
})
RIVER_NAMES = set()
RIVER_BBOX = (4.9, 48.7, 5.9, 49.5)
EXTRA_RIVERS = osmwater.lines('data/osm_verdun/water.json',
                              {"L'Orne", "L'Othain", 'Le Loison', "L'Aire", 'La Creuë', "L'Ezrule", 'La Cousances',
                               'Le Longeau', "L'Andon", 'La Buante', 'La Vadelaincourt', 'Hardillon', 'La Sommedieue',
                               'Ancien Bras de la Meuse'})
WATER_RIVERS = osmwater.lines('data/osm_verdun/water.json', {'La Meuse'})
WATER_HALF_WIDTH = 0.9
LAKES = []
WATER_LEVEL = 0
MIN_WATER_CORNERS, MIN_LAND_CORNERS = 6, 8
MOUNTAIN_TREES = ['TREE05', 'TREE06', 'TREE07', 'TREE08', 'TREE15', 'TREE01', 'TREE02', 'TREE03']
VALLEY_TREES = ['TREE01', 'TREE02', 'TREE03', 'TREE04', 'TREE10', 'TREE11', 'TREE12', 'TREE17']
FRONT_1916 = [  # the front line fought over in 1916 (Argonne - left bank - right bank - Woëvre - Saint-Mihiel)
    [(5.00, 49.195), (5.0767, 49.2012), (5.142, 49.207), (5.17, 49.215), (5.2096, 49.2261), (5.2511, 49.2315),
     (5.28, 49.232), (5.32, 49.237), (5.3816, 49.2250), (5.40, 49.235), (5.4386, 49.2169), (5.4331, 49.1945),
     (5.4702, 49.2005), (5.50, 49.20), (5.51, 49.175), (5.53, 49.155), (5.5603, 49.1157), (5.5811, 49.0525),
     (5.585, 49.00), (5.57, 48.93), (5.556, 48.895), (5.52, 48.885), (5.50, 48.86)],
]


def ground_cover(g):
    E, R, lon, lat, n1, n2 = g.ELEV_CELL, g.RELIEF_CELL, g.CELL_LON, g.CELL_LAT, g.NOISE, g.NOISE2
    cover = np.full(E.shape, g.GRASS)
    p = np.full(E.shape, 0.01)
    front_d = g.polyline_distance(FRONT_1916)
    front = front_d < 5.5 + 2.5 * (n2 - 0.5)                  # the shell-churned belt either side of the line
    north = lat > 49.15                                        # the Verdun sector proper took the heaviest shelling
    heights = (E > 300) | (R > 18)                            # Côtes de Meuse, the Argonne ridges
    argonne = lon < 5.13
    woevre = (lon > 5.55) & (E < 250)
    riparian = g.RIVER_D < 1.3

    cover[n1 > 0.48] = g.THICK                                 # meadows and hay fields
    cover[woevre & (n1 > 0.3)] = g.THICK                       # wet Woëvre pasture
    cover[heights & (n2 > 0.62)] = g.ROUGH                     # limestone outcrops on the Côtes
    p[:] = 0.012
    p[heights] = 0.14 + 0.16 * (1 - n1[heights])               # the Côtes woods (Bois des Caures, Bois Bourrus...)
    p[argonne] = np.maximum(p[argonne], 0.18 + 0.12 * (1 - n2[argonne]))   # the Argonne forest
    p[riparian] = np.maximum(p[riparian], 0.07)
    p[woevre] = np.minimum(p[woevre], 0.02)
    hell = front & north
    cover[front & (n1 > 0.3)] = g.ROUGH                        # crater fields
    cover[hell & (n1 <= 0.3)] = g.SAND                         # churned mud
    p[front] = 0.01                                            # a few shattered stumps
    p[front & ~north] = 0.03
    return cover, heights, p


# --- buildings ---------------------------------------------------------------------------
from regions.pools import POOLS as CIVILIAN_POOLS   # noqa: E402

CIVILIAN_START_CLEAR = 7
PROP_DENSITY = 0.3
CIVILIAN_ZONES = [  # lon, lat, radius (cells), pool, density; the villages on the front were razed and stay empty
    (5.66, 49.24, 3.5, 'village', 0.4),          # Étain's outskirts
    (5.348, 49.1706, 1.6, 'houses', 0.5),        # Thierville
    (5.27, 49.07, 5, 'farm', 0.25),              # rear area along the Voie Sacrée
    (5.20, 49.00, 5, 'farm', 0.25),
    (5.47, 49.11, 5, 'village', 0.3),            # Meuse villages south of Verdun
    (5.62, 49.12, 5, 'farm', 0.25),              # Woëvre farms
    (5.55, 49.32, 5, 'farm', 0.25),              # German rear
    (5.15, 49.36, 4, 'village', 0.3),            # Romagne / Dun, German rear
    (5.12, 49.06, 4, 'village', 0.3),            # Argonne villages south of Clermont
]
CIVILIAN_FILL = ('small', 0.6)
