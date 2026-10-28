"""Codecs for RA2 map packs: IsoMapPack5/PreviewPack (LZO1X) and OverlayPack (Format80)."""
import base64
import ctypes
import struct

_lzo = ctypes.CDLL("/opt/homebrew/opt/lzo/lib/liblzo2.dylib")


def _lzo_compress(data):
    out = ctypes.create_string_buffer(len(data) + len(data) // 16 + 64 + 3)
    out_len = ctypes.c_size_t(0)
    wrk = ctypes.create_string_buffer(16384 * 8)
    r = _lzo.lzo1x_1_compress(data, ctypes.c_size_t(len(data)), out, ctypes.byref(out_len), wrk)
    assert r == 0, r
    return out.raw[:out_len.value]


def _lzo_decompress(data, max_out):
    out = ctypes.create_string_buffer(max_out)
    out_len = ctypes.c_size_t(max_out)
    r = _lzo.lzo1x_decompress_safe(data, ctypes.c_size_t(len(data)), out, ctypes.byref(out_len), None)
    assert r == 0, r
    return out.raw[:out_len.value]


# --- Format80 (LCW) ---------------------------------------------------------

def f80_encode(data):
    """Simple Format80 encoder: fill runs (0xFE) and literal runs (0x80|n)."""
    out = bytearray()
    i, n = 0, len(data)
    lit = bytearray()

    def flush():
        while lit:
            chunk = lit[:63]
            del lit[:63]
            out.append(0x80 | len(chunk))
            out.extend(chunk)

    while i < n:
        j = i
        while j < n and data[j] == data[i] and j - i < 0xFFFF:
            j += 1
        run = j - i
        if run >= 4:
            flush()
            out.append(0xFE)
            out.extend(struct.pack("<H", run))
            out.append(data[i])
            i = j
        else:
            lit.extend(data[i:j])
            i = j
    flush()
    out.append(0x80)  # end marker
    return bytes(out)


def f80_decode(src, out_len):
    out = bytearray()
    i = 0
    while len(out) < out_len:
        c = src[i]
        i += 1
        if c & 0x80 == 0:  # relative copy
            count = (c >> 4) + 3
            rel = ((c & 0x0F) << 8) | src[i]
            i += 1
            pos = len(out) - rel
            for k in range(count):
                out.append(out[pos + k])
        elif c & 0x40 == 0:  # literal
            count = c & 0x3F
            if count == 0:
                break
            out.extend(src[i:i + count])
            i += count
        elif c == 0xFE:
            count, = struct.unpack_from("<H", src, i)
            out.extend(bytes([src[i + 2]]) * count)
            i += 3
        elif c == 0xFF:
            count, pos = struct.unpack_from("<HH", src, i)
            i += 4
            for k in range(count):
                out.append(out[pos + k])
        else:
            count = (c & 0x3F) + 3
            pos, = struct.unpack_from("<H", src, i)
            i += 2
            for k in range(count):
                out.append(out[pos + k])
    return bytes(out)


# --- sectioned packs ----------------------------------------------------------

def pack(data, fmt):
    """fmt 5 = LZO (IsoMapPack5, PreviewPack), 80 = Format80 (OverlayPack/OverlayDataPack)."""
    out = bytearray()
    for off in range(0, len(data), 8192):
        sec = data[off:off + 8192]
        body = _lzo_compress(sec) if fmt == 5 else f80_encode(sec)
        out += struct.pack("<HH", len(body), len(sec)) + body
    return bytes(out)


def unpack(data, fmt):
    out = bytearray()
    i = 0
    while i < len(data):
        size_in, size_out = struct.unpack_from("<HH", data, i)
        i += 4
        body = data[i:i + size_in]
        out += _lzo_decompress(body, size_out) if fmt == 5 else f80_decode(body, size_out)
        i += size_in
    return bytes(out)


def to_ini_lines(raw):
    b64 = base64.b64encode(raw).decode()
    return [b64[k:k + 70] for k in range(0, len(b64), 70)]


def from_ini_section(values):
    return base64.b64decode("".join(values))
