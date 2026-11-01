"""Lay temperate cliff pieces along the edge of a plateau so that every pair of neighbouring cells is one seen
in the official maps (cliff_rules.json, from learn_cliffs.py).

solve(target, band, valid, rng) takes a boolean plateau mask `target` (True = top of the cliff), a mask `band`
of cells free to become cliff (or to move to the other side), and returns {cell: token} for the band, where a
token is 'L' (foot), 'H' (top) or 't.s' (cliff template t, subtile s). Constraint propagation with
backtracking (a small CSP / wave-function-collapse), walking each stretch of the band in order so that a
dead end only undoes nearby choices; cells outside the band keep their target side. Every L cell is cut off
from every H cell by cliff pieces, also diagonally, so the top can only be reached by air.

layout() wraps solve(): where no layout exists (a slot or spur too thin for the pieces), it smooths the outline
around that spot and tries again.
"""
import json
import os

import numpy as np

RULES = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cliff_rules.json')))
DIRS = [tuple(d) for d in RULES['dirs']]
TOK = ['L', 'H'] + sorted(RULES['freq'], key=lambda k: (int(k.split('.')[0]), int(k.split('.')[1])))
IDX = {t: i for i, t in enumerate(TOK)}
HEIGHT = {t: RULES['height'][t] for t in TOK[2:]}
FWD = [[0] * len(TOK) for _ in DIRS]       # FWD[d][a]: tokens allowed at p + d when p holds a
BWD = [[0] * len(TOK) for _ in DIRS]       # BWD[d][b]: tokens allowed at p - d when p holds b
for d, pairs in enumerate(RULES['pairs']):
    for a, b in pairs + [['L', 'L'], ['H', 'H']]:
        if a in IDX and b in IDX:
            FWD[d][IDX[a]] |= 1 << IDX[b]
            BWD[d][IDX[b]] |= 1 << IDX[a]
ALL = (1 << len(TOK)) - 1


def bits(m):
    while m:
        low = m & -m
        yield low.bit_length() - 1
        m ^= low


_memo = {}
FAILS = {}                              # cell -> failed choices (where the search struggled)


def support(table, d, dom):
    key = (id(table), d, dom)
    if key not in _memo:
        s = 0
        for a in bits(dom):
            s |= table[d][a]
        _memo[key] = s
    return _memo[key]


class Unsolvable(Exception):
    def __init__(self, cell):
        super().__init__(f'no cliff layout around cell {cell}')
        self.cell = cell


def solve(target, band, valid, rng, max_steps=50_000, low_only=None):
    """target, band, valid, low_only: numpy bool arrays indexed [x, y]. Cells in low_only stay flat ground at the
    foot (no cliff piece on them). Returns {(x, y): token} over the band."""
    nx, ny = valid.shape
    FAILS.clear()
    dom = {}
    for x, y in zip(*band.nonzero()):
        if valid[x, y]:
            dom[(x, y)] = 1 if low_only is not None and low_only[x, y] else ALL

    def get(c):
        if c in dom:
            return dom[c]
        x, y = c
        if 0 <= x < nx and 0 <= y < ny and valid[x, y]:
            return 1 << (1 if target[x, y] else 0)
        return None                     # off the map: anything goes

    trail = []

    def set_dom(c, m):
        trail.append((c, dom[c]))
        dom[c] = m

    def propagate(queue):
        while queue:
            c = queue.pop()
            m = get(c)
            for d, (dx, dy) in enumerate(DIRS):
                for nb, table in (((c[0] + dx, c[1] + dy), FWD), ((c[0] - dx, c[1] - dy), BWD)):
                    if nb not in dom:
                        if get(nb) is not None and not (get(nb) & support(table, d, m)):
                            return False
                        continue
                    old = dom[nb]
                    new = old & support(table, d, m)
                    if new != old:
                        if not new:
                            return False
                        set_dom(nb, new)
                        queue.append(nb)
        return True

    if not propagate(list(dom)):
        raise Unsolvable(min(dom, key=lambda c: bin(dom[c]).count('1')))
    weight = {i: RULES['freq'].get(t, 0) for i, t in enumerate(TOK)}
    seen = set()
    for seed in sorted(dom, key=lambda c: (c[0] + c[1], c[0])):
        if seed in seen:
            continue
        order, k = [seed], 0             # one connected stretch of cliff band, in walking order
        seen.add(seed)
        while k < len(order):
            x, y = order[k]
            k += 1
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (x + dx, y + dy)
                    if nb in dom and nb not in seen:
                        seen.add(nb)
                        order.append(nb)
        stack = []                      # [cell, remaining choices, trail length before the choice, its place in order]
        steps, pos = 0, 0
        while True:
            while pos < len(order) and not dom[order[pos]] & (dom[order[pos]] - 1):
                pos += 1
            if pos == len(order):
                break
            c = order[pos]
            pref = 1 if target[c] else 0
            choices = sorted(bits(dom[c]), key=lambda i: (i != pref, i < 2, -weight[i] * (0.5 + rng.random())))
            stack.append([c, choices, len(trail), pos])
            while True:
                steps += 1
                cell, choices, mark, p = stack[-1]
                while len(trail) > mark:
                    t, m = trail.pop()
                    dom[t] = m
                if not choices or steps > max_steps:
                    stack.pop()
                    if not stack or steps > max_steps:
                        raise Unsolvable(max(FAILS, key=FAILS.get) if FAILS else cell)
                    continue
                v = choices.pop(0)
                set_dom(cell, 1 << v)
                if propagate([cell]):
                    pos = p
                    break
                FAILS[cell] = FAILS.get(cell, 0) + 1
    return {c: TOK[dom[c].bit_length() - 1] for c in dom}


def _grow(m, k=1):
    for _ in range(k):
        g = m.copy()
        g[1:] |= m[:-1]
        g[:-1] |= m[1:]
        g[:, 1:] |= m[:, :-1]
        g[:, :-1] |= m[:, 1:]
        m = g
    return m


def majority(m, r=2):
    """Each cell takes the majority of its (2r+1)^2 window."""
    p = np.pad(m.astype(int), r)
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    n = 2 * r + 1
    s = c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]
    return s * 2 > n * n


def layout(target, valid, rng, keep_low=None, keep_high=None, band_width=2, attempts=200, log=print):
    """Solve, smoothing the plateau outline around each spot the pieces cannot follow, until it works.
    keep_low: cells that stay flat ground at the foot; keep_high: cells that stay on the plateau side of the
    outline. Returns (final target, {cell: token})."""
    target = target.copy()
    keep_low = np.zeros_like(target) if keep_low is None else keep_low
    keep_high = np.zeros_like(target) if keep_high is None else keep_high
    X, Y = np.meshgrid(np.arange(target.shape[0]), np.arange(target.shape[1]), indexing='ij')
    for attempt in range(attempts):
        edge = _grow(target) & _grow(~target)
        band = _grow(edge, band_width) & valid
        try:
            return target, solve(target, band, valid, rng, low_only=keep_low)
        except Unsolvable as e:
            x, y = e.cell
            r = 3 + attempt // 40          # widen the repair if the same spot keeps failing
            near = np.hypot(X - x, Y - y) <= r
            smooth = majority(target, 2)
            changed = near & (smooth != target) & ~keep_low & ~keep_high
            if not changed.any():          # nothing to smooth: shave the spot to the side with more room
                changed = near & target & ~keep_high if target[near].mean() < 0.5 else near & ~target & ~keep_low
                target[changed] = ~target[changed]
            else:
                target[changed] = smooth[changed]
            log(f'  cliff: reshaped {int(changed.sum())} cells around {x},{y}')
    raise RuntimeError('cliff layout failed')
