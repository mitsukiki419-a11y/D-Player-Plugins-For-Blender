"""Read / write D-Player 2.0 stage files (.dsz).

Container
---------
A .dsz file is a zlib stream that D-Player "scrambles" when the file is
bigger than a few hundred bytes:

    file = scrambled(zlib_data) + bytes([K])

where the first K bytes and the last K bytes of the zlib data are swapped and
each block is reversed.  Small files (like community-made "cute.dsz") are
plain zlib with no trailer.  Our writer always pads the zlib data to at least
2 KB (zlib ignores trailing bytes) and scrambles it, which is what D-Player
expects for normal-sized stages.

Payload (little endian, version 0x81 - what we write)
-----------------------------------------------------
  u8   version (0x81)
  [version >= 0x83: u32 len + len bytes  (licence tag, e.g. "Charged")]
  u32  n_coords (= 3 * n_verts)
       0x81-0x84: i16 x,y,z   0x85-0x87: i32 x,y,z   0x88: f32 x,y,z
  [version >= 0x86: n_coords * f32 normals]
  [version >= 0x88: n_verts * u16]
  u32  n_uv (= 2 * n_verts)    then f32 u, v per vertex  (v = 0 at the bottom)
  u32  n_extra                 then n_extra * 4 bytes (unused, 0)
  u16  n_materials
     per material: u16 flags(0x1a), u16 name_len, name, u32 0,
                   u32 n_indices, n_indices * u16 (triangle list, CCW = front)
  u16  n_verts   per vertex: u8 count, count * (u8 material, u16 triangle)
                 (faces touching that vertex position)
  u32  n_edges   per edge: u16 v0, u16 v1, u8 open, u8 matA, u16 triA, u8 matB, u16 triB
                 (every edge by position; open=255 when only one face uses it)
  u16 settings_len (0 = none) + 4 zero bytes

Units are millimetres-ish: the floor is y = 0, +y is up, the character
stands at the origin facing +z (towards the default camera).
"""
import struct
import zlib

VERSION = 0x81
TAIL = bytes(6)   # 'no extra settings' block, as in iDance's garage.dsz / mgs.dsz
MAT_FLAGS = 0x1a
MIN_ZLIB_SIZE = 2048
SCRAMBLE_K = 37


# ---------------------------------------------------------------- container
def unscramble(data):
    k = data[-1]
    g = data[:-1]
    n = len(g)
    if k == 0 or 2 * k > n:
        raise ValueError('bad scramble trailer')
    d = bytearray(g)
    d[0:k] = g[n - k:n][::-1]
    d[n - k:n] = g[0:k][::-1]
    return bytes(d)


def scramble(zdata, k=SCRAMBLE_K):
    if len(zdata) < MIN_ZLIB_SIZE:
        zdata = zdata + bytes(MIN_ZLIB_SIZE - len(zdata))
    n = len(zdata)
    d = bytearray(zdata)
    d[0:k] = zdata[n - k:n][::-1]
    d[n - k:n] = zdata[0:k][::-1]
    return bytes(d) + bytes([k])


def decode_file(data):
    """Return the raw payload of a .dsz file (either container style)."""
    if data[:1] == b'\x78':
        try:
            return zlib.decompressobj().decompress(data)
        except zlib.error:
            pass
    o = zlib.decompressobj()
    return o.decompress(unscramble(data))


def encode_file(raw):
    return scramble(zlib.compress(raw, 9))


# ---------------------------------------------------------------- writer
def _key(p):
    return (int(round(p[0])), int(round(p[1])), int(round(p[2])))


def build(verts, uvs, materials, tail=TAIL):
    """verts: [(x,y,z)], uvs: [(u,v)], materials: [(texture_path, [indices])]"""
    if len(verts) != len(uvs):
        raise ValueError('verts/uvs length mismatch')
    if len(verts) > 65535:
        raise ValueError('too many vertices (max 65535) - split or decimate the stage')
    if len(materials) > 255:
        raise ValueError('too many materials (max 255)')
    out = bytearray()
    out += struct.pack('<BI', VERSION, len(verts) * 3)
    for v in verts:
        out += struct.pack('<3h', *[max(-32768, min(32767, int(round(c)))) for c in v])
    out += struct.pack('<I', len(uvs) * 2)
    for uv in uvs:
        out += struct.pack('<2f', float(uv[0]), float(uv[1]))
    out += struct.pack('<I', 0)
    out += struct.pack('<H', len(materials))
    for name, idx in materials:
        if len(idx) // 3 > 65535:
            raise ValueError('material %s has too many triangles' % name)
        nb = name.encode('cp949', 'replace')
        out += struct.pack('<HH', MAT_FLAGS, len(nb)) + nb
        out += struct.pack('<II', 0, len(idx))
        out += struct.pack('<%dH' % len(idx), *idx)

    faces_at = {}
    for mi, (_, idx) in enumerate(materials):
        for t in range(len(idx) // 3):
            for vi in idx[3 * t:3 * t + 3]:
                lst = faces_at.setdefault(_key(verts[vi]), [])
                if (mi, t) not in lst:
                    lst.append((mi, t))
    out += struct.pack('<H', len(verts))
    for v in verts:
        lst = faces_at.get(_key(v), [])[:255]
        out += struct.pack('<B', len(lst))
        for mi, t in lst:
            out += struct.pack('<BH', mi, t)

    edge_faces, edge_rep, order = {}, {}, []
    for mi, (_, idx) in enumerate(materials):
        for t in range(len(idx) // 3):
            tri = idx[3 * t:3 * t + 3]
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                k = frozenset((_key(verts[a]), _key(verts[b])))
                if len(k) < 2:
                    continue
                if k not in edge_faces:
                    edge_faces[k] = []
                    edge_rep[k] = (a, b)
                    order.append(k)
                edge_faces[k].append((mi, t))
    out += struct.pack('<I', len(order))
    for k in order:
        a, b = edge_rep[k]
        fl = edge_faces[k]
        if len(fl) >= 2:
            (ma, ta), (mb, tb) = fl[0], fl[1]
            out += struct.pack('<HHBBHBH', a, b, 0, ma, ta, mb, tb)
        else:
            ma, ta = fl[0]
            out += struct.pack('<HHBBHBH', a, b, 255, ma, ta, 0, 0)
    out += tail
    return bytes(out)


def save(path, verts, uvs, materials, **kw):
    with open(path, 'wb') as f:
        f.write(encode_file(build(verts, uvs, materials, **kw)))


# ---------------------------------------------------------------- reader
def parse(raw):
    p = 0
    ver = raw[0]
    p = 1
    if ver >= 0x83:
        ln = struct.unpack_from('<I', raw, p)[0]
        p += 4 + ln
    n = struct.unpack_from('<I', raw, p)[0]
    p += 4
    nv = n // 3
    if ver >= 0x88:
        verts = [struct.unpack_from('<3f', raw, p + 12 * i) for i in range(nv)]
        p += 12 * nv
    elif ver >= 0x85:
        verts = [struct.unpack_from('<3i', raw, p + 12 * i) for i in range(nv)]
        p += 12 * nv
    else:
        verts = [struct.unpack_from('<3h', raw, p + 6 * i) for i in range(nv)]
        p += 6 * nv
    if ver >= 0x86:
        p += 4 * n
    if ver >= 0x88:
        p += (4 * n // 3) // 2
    m = struct.unpack_from('<I', raw, p)[0]
    p += 4
    uvs = [struct.unpack_from('<2f', raw, p + 8 * i) for i in range(m // 2)]
    p += 4 * m
    k = struct.unpack_from('<I', raw, p)[0]
    p += 4 + 4 * k
    nm = struct.unpack_from('<H', raw, p)[0]
    p += 2
    mats = []
    for _ in range(nm):
        _fl, sl = struct.unpack_from('<HH', raw, p)
        p += 4
        name = raw[p:p + sl].decode('cp949', 'replace')
        p += sl
        _z, ni = struct.unpack_from('<II', raw, p)
        p += 8
        idx = list(struct.unpack_from('<%dH' % ni, raw, p))
        p += 2 * ni
        mats.append((name, idx))
    if len(uvs) < len(verts):
        uvs += [(0.0, 0.0)] * (len(verts) - len(uvs))
    return dict(version=ver, verts=verts, uvs=uvs[:len(verts)], materials=mats)


def load(path):
    with open(path, 'rb') as f:
        return parse(decode_file(f.read()))
