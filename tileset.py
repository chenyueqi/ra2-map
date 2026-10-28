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
