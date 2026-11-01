"""Parse OpenRA ra2 temperat.yaml: per-template category, size, per-tile ramp/height/terrain/colors."""
import re

def load(path='ref/temperat.yaml'):
    txt = open(path).read()
    out = {}
    for b in re.split(r'\n\tTemplate@', txt)[1:]:
        tid = int(re.search(r'\n\t\tId: (\d+)', b).group(1))
        cat = re.search(r'Categories: (.*)', b).group(1)
        size = tuple(int(v) for v in re.search(r'Size: (.*)', b).group(1).split(','))
        tiles = {}
        for m in re.finditer(r'\n\t\t\t(\d+): (\w+)((?:\n\t\t\t\t.*)*)', b):
            props = dict(re.findall(r'\n\t\t\t\t(\w+): (.*)', m.group(3)))
            tiles[int(m.group(1))] = dict(terrain=m.group(2), ramp=int(props.get('RampType', 0)),
                                          height=int(props.get('Height', 0)),
                                          color=props.get('MinColor'), color2=props.get('MaxColor'))
        out[tid] = dict(cat=cat, size=size, tiles=tiles)
    return out


THEATER_FILES = {'TEMPERATE': 'temperat.yaml', 'URBAN': 'urban.yaml', 'SNOW': 'snow.yaml'}
# ground covers as (base tile, first of its 16 LAT transition tiles); checked against real maps with learn_theater.py
_TEM = dict(THICK=(131, 132), ROUGH=(418, 419), SAND=(493, 494), PAVE=(534, 463))
GROUND = {'TEMPERATE': _TEM, 'URBAN': _TEM,
          'SNOW': dict(THICK=(131, 132),        # bare grass showing through the snow (fields, reeds)
                       ROUGH=(696, 697),        # snowy rock
                       SAND=(131, 132),         # no beach sand under snow: frozen grass
                       PAVE=(734, 749))}
