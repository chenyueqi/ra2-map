"""Lakes and reservoirs from an OpenStreetMap Overpass dump (out geom): named polygons -> water corners."""
import json
import os

import numpy as np
from matplotlib.path import Path

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def _stitch(ways):
    """Join open member ways of a multipolygon into closed rings."""
    ways = [list(w) for w in ways if len(w) > 1]
    rings = []
    while ways:
        ring = ways.pop()
        while ring[0] != ring[-1]:
            for i, w in enumerate(ways):
                if w[0] == ring[-1]:
                    ring += w[1:]
                elif w[-1] == ring[-1]:
                    ring += w[-2::-1]
                else:
                    continue
                ways.pop(i)
                break
            else:
                break                      # cut by the query bbox: close it straight
        rings.append(ring)
    return rings


def polygons(path, names):
    """Outer rings (lists of (lon, lat)) of the named water polygons in an Overpass json dump."""
    rings = []
    for e in json.load(open(os.path.join(HERE, path)))['elements']:
        if e.get('tags', {}).get('name') not in names:
            continue
        if e['type'] == 'way':
            rings.append([(p['lon'], p['lat']) for p in e['geometry']])
        elif e['type'] == 'relation':
            outer = [[(p['lon'], p['lat']) for p in m['geometry']] for m in e['members']
                     if m.get('role') == 'outer' and 'geometry' in m]
            rings += _stitch(outer)
    return rings


def ne_lakes(names, path='data/lakes.geojson'):
    """Outer rings of the named Natural Earth lakes."""
    rings = []
    for f in json.load(open(os.path.join(HERE, path)))['features']:
        if f['properties'].get('name') in names:
            g = f['geometry']
            rings += [p[0] for p in g['coordinates']] if g['type'] == 'MultiPolygon' else [g['coordinates'][0]]
    return [[tuple(p) for p in r] for r in rings]


def lines(path, names):
    """Polylines of the named waterways (rivers, canals)."""
    return [[(p['lon'], p['lat']) for p in e['geometry']]
            for e in json.load(open(os.path.join(HERE, path)))['elements']
            if e['type'] == 'way' and 'waterway' in e.get('tags', {}) and e['tags'].get('name') in names]


def mask(g, rings, grow=0):
    """Water corners inside the rings, dilated by `grow` corners so narrow reservoirs survive the 2x2 shore blocks."""
    pts = np.column_stack([g.CORNER_LON.ravel(), g.CORNER_LAT.ravel()])
    wet = np.zeros(len(pts), bool)
    for r in rings:
        if len(r) > 2:
            wet |= Path(r).contains_points(pts)
    wet = wet.reshape(g.CORNER_LON.shape)
    for _ in range(grow):
        wet = wet | g.shift(wet, 1, 0, False) | g.shift(wet, -1, 0, False) | g.shift(wet, 0, 1, False) | g.shift(wet, 0, -1, False)
    return wet
