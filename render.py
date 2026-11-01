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
                               tileset.THEATER_FILES[obj.get('theater', 'TEMPERATE')]))
h, level, tile, sub, overlay, valid = st['heights'], st['level'], st['tile'], st['sub'], st['overlay'], st['valid']
N = valid.shape[0]
if 'corners' in st:                     # each cell's own corner heights (cliffs break the shared corner grid)
    ch = st['corners']
else:
    ch = np.stack([h[:N, :N], np.roll(h, -1, 0)[:N, :N], np.roll(np.roll(h, -1, 0), -1, 1)[:N, :N],
                   np.roll(h, -1, 1)[:N, :N]], axis=-1)
cliff = st['cliff'] if 'cliff' in st else np.zeros_like(valid)
TOP = max(6, int(level[valid].max()) // 2 + 3)          # room above the first row for raised ground
W, H = obj['W'], obj['H']
LX, LY, LW, LH = obj['local']
X0, X1, Y0, Y1 = -(W - 2 * LX), 2 * (LX + LW) - W, W + 2 * LY, W + 2 * (LY + LH + 1)


def pt(cx, cy, z=None):
    return (cx - cy - X0) * S, (cx + cy - Y0) * S / 2 - (h[cx, cy] if z is None else z) * S / 2 + TOP * S


img = Image.new('RGB', ((X1 - X0) * S, int((Y1 - Y0) * S / 2 + (TOP + 2) * S)), (20, 20, 24))
d = ImageDraw.Draw(img)
cells = sorted(zip(*np.nonzero(valid)), key=lambda c: (c[0] + c[1], c[0]))
for rx, ry in cells:
    t = int(tile[rx, ry])
    col = np.array([int(TS[t]['tiles'][int(sub[rx, ry])]['color'][i:i + 2], 16) for i in (0, 2, 4)], float)
    c = [int(v) for v in ch[rx, ry]]
    shade = 1.0 + 0.16 * ((c[2] + c[1]) - (c[0] + c[3])) + 0.025 * level[rx, ry]
    if 'bridge' in st and st['bridge'][rx, ry]:
        col = np.array([150, 146, 136.])
    elif overlay[rx, ry] != 255:
        col = np.array([140, 122, 230.]) if overlay[rx, ry] < 100 else np.array([222, 180, 50.])
    col = tuple(int(v) for v in np.clip(col * shade, 0, 255))
    poly = [pt(rx, ry, c[0]), pt(rx + 1, ry, c[1]), pt(rx + 1, ry + 1, c[2]), pt(rx, ry + 1, c[3])]
    d.polygon(poly, fill=col)
    for (dx, dy), (a, b), dark in (((1, 0), (1, 2), 0.55), ((0, 1), (2, 3), 0.75)):   # rock face down to a lower cell
        nb = (rx + dx, ry + dy)
        if nb[0] < N and nb[1] < N and valid[nb]:
            drop = min(c[a], c[b]) - max(ch[nb][3 if dx else 0], ch[nb][0 if dx else 1])
            if drop >= 3 or (cliff[rx, ry] and drop > 0):
                ea, eb = (rx + 1, ry) if dx else (rx + 1, ry + 1), (rx + 1, ry + 1) if dx else (rx, ry + 1)
                low = int(level[nb])
                rock = tuple(int(v * dark) for v in (150, 135, 115))
                d.polygon([pt(*ea, c[a]), pt(*eb, c[b]), pt(*eb, low), pt(*ea, low)], fill=rock)
for k, name in obj['terrain'].items():
    k = int(k)
    rx, ry = k % 1000, k // 1000
    x, y = (rx - ry - X0) * S, (rx + ry + 1 - Y0) * S / 2 - level[rx, ry] * S / 2 + TOP * S
    if name.startswith('POLE'):                         # telegraph pole: a thin dark post
        d.line([x, y - S * 1.4, x, y], fill=(60, 45, 35), width=max(1, S // 4))
        continue
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
    x, y = (rx - ry - X0) * S, (rx + ry + 1 - Y0) * S / 2 - level[rx, ry] * S / 2 + TOP * S
    r = S * 1.6
    d.ellipse([x - r, y - r, x + r, y + r], fill=colors[i], outline=(255, 255, 255), width=2)
    label = f'{i + 1} {name}'
    tw = d.textlength(label, font=font)
    tx = x + r + 4 if x + r + 4 + tw < img.width else x - r - 4 - tw   # near the right edge: label on the left
    d.text((tx, y - r), label, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
img.save(os.path.join(OUT, f'render_{S}.png'))
print(img.size)
