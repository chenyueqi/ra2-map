"""Isometric overview render of the generated map (for checking, not used by the game)."""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import tileset

OUT = os.path.join(os.path.dirname(__file__), 'out', sys.argv[1] if len(sys.argv) > 1 else 'colorado')
S = int(sys.argv[2]) if len(sys.argv) > 2 else 8          # half cell width in pixels
st = np.load(os.path.join(OUT, 'state.npz'))
obj = json.load(open(os.path.join(OUT, 'objects.json')))
TS = tileset.load(os.path.join(os.path.dirname(__file__), 'ref',
                               {'TEMPERATE': 'temperat.yaml', 'URBAN': 'urban.yaml'}[obj.get('theater', 'TEMPERATE')]))
h, level, tile, sub, overlay, valid = st['heights'], st['level'], st['tile'], st['sub'], st['overlay'], st['valid']
W, H = obj['W'], obj['H']
LX, LY, LW, LH = obj['local']
X0, X1, Y0, Y1 = -(W - 2 * LX), 2 * (LX + LW) - W, W + 2 * LY, W + 2 * (LY + LH + 1)


def pt(cx, cy):
    return (cx - cy - X0) * S, (cx + cy - Y0) * S / 2 - h[cx, cy] * S / 2 + 6 * S


img = Image.new('RGB', ((X1 - X0) * S, int((Y1 - Y0) * S / 2 + 8 * S)), (20, 20, 24))
d = ImageDraw.Draw(img)
cells = sorted(zip(*np.nonzero(valid)), key=lambda c: (c[0] + c[1], c[0]))
for rx, ry in cells:
    t = int(tile[rx, ry])
    col = np.array([int(TS[t]['tiles'][int(sub[rx, ry])]['color'][i:i + 2], 16) for i in (0, 2, 4)], float)
    c = [h[rx, ry], h[rx + 1, ry], h[rx + 1, ry + 1], h[rx, ry + 1]]
    shade = 1.0 + 0.16 * ((c[2] + c[1]) - (c[0] + c[3])) + 0.025 * level[rx, ry]
    if 'bridge' in st and st['bridge'][rx, ry]:
        col = np.array([150, 146, 136.])
    elif overlay[rx, ry] != 255:
        col = np.array([140, 122, 230.]) if overlay[rx, ry] < 100 else np.array([222, 180, 50.])
    col = tuple(int(v) for v in np.clip(col * shade, 0, 255))
    poly = [pt(rx, ry), pt(rx + 1, ry), pt(rx + 1, ry + 1), pt(rx, ry + 1)]
    d.polygon(poly, fill=col)
for k, name in obj['terrain'].items():
    k = int(k)
    rx, ry = k % 1000, k // 1000
    x, y = (rx - ry - X0) * S, (rx + ry + 1 - Y0) * S / 2 - level[rx, ry] * S / 2 + 6 * S
    col = (34, 70, 30) if name.startswith('TREE') else (230, 120, 20)
    d.ellipse([x - S * 0.55, y - S * 0.9, x + S * 0.55, y + S * 0.2], fill=col)
import buildings
CIV = buildings.load(obj.get('theater', 'TEMPERATE'))
DIMS = {'CAOILD': (2, 2), 'CAAIRP': (3, 3), 'CAHOSP': (6, 4), 'CAOUTP': (4, 3)}
for s in obj['structures']:
    f = s.split(',')
    rx, ry, kind = int(f[3]), int(f[4]), f[1]
    fw, fh = DIMS.get(kind) or (CIV[kind]['w'], CIV[kind]['h'])
    tall = S * (2.5 if kind.startswith(('CANEWY', 'CANWY', 'CACHIG', 'CAMIAM', 'CAWASH')) else 1.2)
    base = [pt(rx, ry), pt(rx + fw, ry), pt(rx + fw, ry + fh), pt(rx, ry + fh)]
    top = [(x, y - tall) for x, y in base]
    civ = kind not in DIMS
    wall, roof = ((150, 140, 130), (205, 195, 180)) if civ else ((80, 80, 95), (120, 120, 140))
    d.polygon([base[1], base[2], top[2], top[1]], fill=tuple(int(v * 0.8) for v in wall))
    d.polygon([base[2], base[3], top[3], top[2]], fill=wall)
    d.polygon(top, fill=roof, outline=(60, 60, 60))
try:
    font = ImageFont.truetype('/System/Library/Fonts/Hiragino Sans GB.ttc', max(12, int(S * 1.6)))
except OSError:
    font = ImageFont.load_default()
colors = [(220, 40, 40), (40, 110, 230), (240, 200, 30), (40, 180, 60), (240, 130, 20), (150, 60, 200),
          (40, 200, 200), (230, 90, 170)]
for i, ((rx, ry), name) in enumerate(zip(obj['starts'], obj['names'])):
    x, y = (rx - ry - X0) * S, (rx + ry + 1 - Y0) * S / 2 - level[rx, ry] * S / 2 + 6 * S
    r = S * 1.6
    d.ellipse([x - r, y - r, x + r, y + r], fill=colors[i], outline=(255, 255, 255), width=2)
    d.text((x + r + 4, y - r), f'{i + 1} {name}', fill=(255, 255, 255), font=font, stroke_width=2,
           stroke_fill=(0, 0, 0))
img.save(os.path.join(OUT, f'render_{S}.png'))
print(img.size)
