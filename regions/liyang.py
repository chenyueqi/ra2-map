"""Liyang (溧阳): 8 players around the neutral city; Nanshan bamboo hills, Tianmu Lake, Changdang Lake, Wawu Shan."""
import numpy as np

from regions import osmwater

NAME = 'Liyang'
OUTFILE = 'liyang.mpr'
SEED = 1993  # Liyang county became a city
W, H = 131, 148
LX, LY, LW, LH = 2, 4, 127, 135
LON0, LON1, LAT0, LAT1 = 119.08, 119.70, 31.70, 31.13
DEM_DIR, DEM_ZOOM = 'data/dem_liyang', 10
BLUR_SMOOTH, BLUR_WIDE, BLUR_TERRAIN = 2, 7, 3
ELEV_PTS = [0, 22, 45, 80, 125, 180, 250, 330, 430, 560]
LVL_PTS = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8.2]
LIGHTING = dict(Red=0.98, Green=1.0, Blue=0.97, Level=0.03, Ambient=1.02)

STARTS = [
    ('上兴', 119.24, 31.55),
    ('社头', 119.42, 31.65),
    ('上黄', 119.60, 31.52),
    ('芳庄', 119.63, 31.40),
    ('戴埠', 119.50, 31.31),
    ('南山竹海', 119.42, 31.18),
    ('社渚', 119.27, 31.32),
    ('桠溪', 119.15, 31.43),
]
START_FLAT, START_ORE_DIST = 11, 8.5
MAP_CENTER = (119.48, 31.42)
ORE_TOWNS = [('竹箦', 119.33, 31.56), ('白马', 119.17, 31.62), ('别桥', 119.42, 31.555), ('埭头', 119.52, 31.50),
             ('南渡', 119.33, 31.44), ('古县', 119.45, 31.36), ('梅渚', 119.21, 31.25), ('指前', 119.47, 31.61),
             ('西渚', 119.58, 31.28), ('凌笪', 119.30, 31.20)]
GEM_TOWNS = [('瓦屋山', 119.27, 31.66), ('曹山', 119.56, 31.37), ('龙潭林场', 119.58, 31.19)]
OIL = [('昆仑开发区', 119.40, 31.50), ('溧阳东', 119.56, 31.43), ('上兴南', 119.22, 31.46), ('社渚东', 119.35, 31.27)]
TECH = [('CAAIRP', '别桥通用机场', 119.43, 31.52, 3, 3),
        ('CAHOSP', '溧阳市人民医院', 119.48, 31.43, 6, 4),
        ('CAOUTP', '溧阳高铁站', 119.46, 31.38, 4, 3)]
CITIES = [('溧阳', 119.48, 31.42, 7), ('昆仑', 119.45, 31.45, 4), ('天目湖', 119.44, 31.33, 2), ('戴埠', 119.50, 31.31, 1.5),
          ('上黄', 119.60, 31.52, 1.5), ('别桥', 119.46, 31.56, 1.5), ('社渚', 119.27, 31.32, 1.5), ('上兴', 119.24, 31.55, 1.5),
          ('南渡', 119.33, 31.44, 1.5), ('竹箦', 119.33, 31.56, 1.2), ('埭头', 119.52, 31.50, 1.2)]
ROADS = {
    'G25 长深高速': [(119.38, 31.70), (119.44, 31.52), (119.47, 31.42), (119.50, 31.30), (119.55, 31.13)],
    'G104': [(119.08, 31.50), (119.24, 31.55), (119.33, 31.44), (119.48, 31.42), (119.63, 31.40), (119.70, 31.38)],
    '常溧高速': [(119.48, 31.42), (119.60, 31.52), (119.70, 31.58)],
    '溧高高速': [(119.48, 31.42), (119.27, 31.32), (119.08, 31.30)],
    '溧金': [(119.48, 31.42), (119.46, 31.56), (119.45, 31.70)],
    '天目湖': [(119.48, 31.42), (119.44, 31.33), (119.42, 31.18)],
}
RIVER_NAMES = set()
RIVER_BBOX = (118.9, 31.0, 119.9, 31.9)
EXTRA_RIVERS = osmwater.lines('data/osm_liyang/water.json',
                              {'南河', '中河', '北河', '丹金溧漕河', '芜申运河', '胥河', '郎川河', '新郎川河', '大溪河',
                               '戴溧河', '竹箦河', '上兴河', '梅渚河', '濑溪河'})
WATER_RIVERS, WATER_HALF_WIDTH, LAKES = [], 0, []
WATER_LEVEL = 0
MIN_WATER_CORNERS, MIN_LAND_CORNERS = 4, 4
LAKES_BIG = {'长荡湖', '方便水库'}
LAKES_NARROW = {'沙河水库（天目湖）', '大溪水库', '横山水库'}   # widened a little to stay visible at 650 m a cell
MOUNTAIN_TREES = ['TREE25', 'TREE26', 'TREE27', 'TREE05', 'TREE06', 'TREE07', 'TREE08', 'TREE15']
VALLEY_TREES = ['TREE01', 'TREE02', 'TREE03', 'TREE04', 'TREE10', 'TREE11', 'TREE12', 'TREE17']


def water_corners(g):
    src = 'data/osm_liyang/water.json'
    return osmwater.mask(g, osmwater.polygons(src, LAKES_BIG)) | osmwater.mask(g, osmwater.polygons(src, LAKES_NARROW), grow=1)


def ground_cover(g):
    E, R, lon, lat, n1, n2 = g.ELEV_CELL, g.RELIEF_CELL, g.CELL_LON, g.CELL_LAT, g.NOISE, g.NOISE2
    cover = np.full(E.shape, g.GRASS)
    p = np.full(E.shape, 0.012)
    hills = (E > 40) | (R > 25)
    plain = ~hills
    bamboo = hills & (lat < 31.34)                              # 南山 bamboo sea and the Tianmu hills
    wawu = hills & (lat > 31.5) & (lon < 119.36)                # 瓦屋山 / 茅山 foothills
    riparian = (g.RIVER_D < 1.2) & plain

    cover[plain & (n1 > 0.58)] = g.THICK                        # rice paddies
    cover[plain & (n2 > 0.8) & (n1 <= 0.58)] = g.ROUGH          # threshing grounds, fish-pond dikes
    cover[hills & (E < 120) & (n2 > 0.55)] = g.THICK            # tea gardens on the lower slopes (白茶)
    cover[hills & (E >= 250) & (n1 > 0.6)] = g.ROUGH            # bare ridge tops
    p[hills] = 0.05
    p[bamboo] = 0.22 + 0.3 * (1 - n1[bamboo])
    p[wawu] = 0.12 + 0.2 * (1 - n1[wawu])
    p[riparian] = 0.18
    return cover, hills, p


# --- buildings ---------------------------------------------------------------------------
from regions.pools import POOLS as CIVILIAN_POOLS   # noqa: E402

CIVILIAN_START_CLEAR = 10
FLAT_CITIES = [('溧阳', 119.48, 31.42, 6)]
PROP_DENSITY = 0.3
CIVILIAN_ZONES = [  # lon, lat, radius (cells), pool, density
    (119.48, 31.425, 5, 'highrise', 1.0),        # 溧城 old town / 燕山
    (119.455, 31.45, 4, 'highrise', 0.9),        # 昆仑 new district
    (119.43, 31.47, 3, 'industrial', 0.6),       # 昆仑开发区
    (119.52, 31.40, 3, 'suburb', 0.5),
    (119.44, 31.335, 2.5, 'suburb', 0.5),        # 天目湖 resort town
    (119.36, 31.48, 5, 'farm', 0.25),            # paddy villages of the central plain
    (119.55, 31.46, 4, 'farm', 0.25),
    (119.30, 31.38, 4, 'farm', 0.25),
    (119.40, 31.58, 4, 'village', 0.35),
    (119.20, 31.50, 3, 'village', 0.35),
    (119.55, 31.33, 3, 'village', 0.35),
    (119.33, 31.25, 3, 'village', 0.3),
    (119.60, 31.45, 3, 'village', 0.35),
]
CIVILIAN_FILL = ('fill', 0.9)
