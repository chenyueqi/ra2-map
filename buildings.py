"""RA2 civilian buildings usable in the TEMPERATE theater, with footprints (from the OpenRA ra2 mod data)."""
import os
import re

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ref')


def _blocks(path):
    txt = open(path).read()
    return {m.group(1): m.group(2) for m in re.finditer(r'^([A-Za-z0-9_.\-]+):\n((?:[\t ].*\n|\n)*)', txt, re.M)}


def load(theater='TEMPERATE'):
    rules = _blocks(os.path.join(REF, 'civilian-structures.yaml'))
    seqs = _blocks(os.path.join(REF, 'seq-civilian-structures.yaml'))
    out = {}
    for name, body in rules.items():
        if '.' in name or name.startswith('^'):
            continue
        dims = re.search(r'Dimensions:\s*(\d+)\s*,\s*(\d+)', body)
        if not dims or 'Building:' not in body:
            continue
        img = re.search(r'RenderSprites:\s*\n\t\tImage:\s*(\S+)', body)
        seq = seqs.get(img.group(1) if img else name, '')
        fn = re.search(r'\n\t\tFilename:\s*(\S+)', seq)
        # art is safe in a theater if it has a variant for it or lives in the generic ("cg..."/"ng...") mix
        available = f'{theater}:' in seq or bool(fn and fn.group(1)[1] == 'g')
        tip = re.search(r'Tooltip:\s*\n\t\tName:\s*(.*)', body)
        out[name.upper()] = dict(w=int(dims.group(1)), h=int(dims.group(2)), available=available,
                                 name=tip.group(1).strip() if tip else '', walk='Footprint: =' in body)
    return out


if __name__ == '__main__':
    import sys
    theater = sys.argv[1] if len(sys.argv) > 1 else 'TEMPERATE'
    B = load(theater)
    t = {k: v for k, v in B.items() if v['available']}
    print(len(B), 'buildings,', len(t), f'with {theater} art')
    for k, v in sorted(t.items()):
        print(f"{k:12s} {v['w']}x{v['h']}  {v['name']}")
