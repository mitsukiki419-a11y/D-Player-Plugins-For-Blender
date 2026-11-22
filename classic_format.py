"""D-Player Tools by ChiakiMK.

Read / write classic D-Player characters (.dcz), motion pools (.dmp) and dance routines (.dsq).

Pure Python (no bpy) so it can be tested outside Blender.

Coordinates are D-Player's: millimetres, +Y up, the character faces +Z, its left hand is +X.

.dcz (scrambled zlib, like .dsz)
    u8 0x81; u32 3n; n * i16 xyz  (vertex position relative to its bone, mm, bone rest = no rotation)
    u32 n; n * u8 weight (127 = all own bone, the rest goes to the bone's parent)
    u32 2n; n * f32 u,v;  u32 0;  u16 n_mat; materials (u16 0x1a, u16 len, name, u32 0, u32 n_idx, u16 idx...)
    u16 3*nb; nb * (i16 x,y,z offset from parent, i16 parent)
    nb * (u16 count, count * u16 vertex)
    adjacency + edges (same as stages), 6 zero bytes

.dmp  "MP" u8 1, u32 id, u16 n, n * u32 record size;
      record = u16 index, "MM", u32 size, scrambled zlib of:
      u8 segments, u16 frames, frames * i16 root xyz, frames * 19 * i16 quaternion xyzw * 10000  (30 fps)

.dsq  zlib of blocks: "SQ" 01 u16 n | "SF" u8 len sound | "MS" 01 u16 n u8 dancers, pool[n], seg[n], u16 np,
      np * (u32 pool id, u16 motion) | "CS" camera (same shape + span[n]) | "BS" 01 u16 start u16 n,
      per slot u8 c + c * u16 beat ms | "SE"
"""
import math
import os
import random
import struct
import zlib

from . import dsz_format as dz

CLASSIC_BONES = [
    'Hips', 'Spine', 'Spine2', 'Neck', 'Head',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    'LeftUpLeg', 'LeftLeg', 'LeftFoot',
    'RightUpLeg', 'RightLeg', 'RightFoot',
]
PARENTS = [-1, 0, 1, 2, 3, 2, 5, 6, 7, 2, 9, 10, 11, 0, 13, 14, 0, 16, 17]
ROOT_REF = 2419.0      # hips height in motion files when standing
HIPS_MM = 1100.0       # hips height of the reference skeleton those motions were made on
FPS = 30


# ---------------------------------------------------------------- quaternion helpers (x, y, z, w)
def q_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def q_inv(a):
    return (-a[0], -a[1], -a[2], a[3])


def q_norm(a):
    n = math.sqrt(sum(c * c for c in a)) or 1.0
    return tuple(c / n for c in a)


def q_slerp(a, b, t):
    d = sum(x * y for x, y in zip(a, b))
    if d < 0:
        b = tuple(-c for c in b)
        d = -d
    if d > 0.9995:
        return q_norm(tuple(x + (y - x) * t for x, y in zip(a, b)))
    th = math.acos(min(1.0, d))
    s = math.sin(th)
    wa, wb = math.sin((1 - t) * th) / s, math.sin(t * th) / s
    return tuple(wa * x + wb * y for x, y in zip(a, b))


# ---------------------------------------------------------------- containers
def decode(data):
    if data[:1] == b'\x78':
        try:
            return zlib.decompressobj().decompress(data)
        except zlib.error:
            pass
    try:
        return zlib.decompressobj().decompress(dz.unscramble(data))
    except Exception:
        return zlib.decompressobj(-15).decompress(dz.unscramble(data)[2:])


def encode(raw):
    return dz.encode_file(raw)


# ---------------------------------------------------------------- .dcz
def _parse_dcz_at(raw, p):
    n = struct.unpack_from('<I', raw, p)[0]; p += 4
    if n == 0 or n % 3 or n > 600000:
        raise ValueError('vertex count')
    nv = n // 3
    local = [struct.unpack_from('<3h', raw, p + 6 * i) for i in range(nv)]; p += 6 * nv
    c = struct.unpack_from('<I', raw, p)[0]; p += 4
    if c != nv:
        raise ValueError('weights')
    weights = list(raw[p:p + c]); p += c
    m = struct.unpack_from('<I', raw, p)[0]; p += 4
    if m != 2 * nv:
        raise ValueError('uvs')
    uvs = [struct.unpack_from('<2f', raw, p + 8 * i) for i in range(nv)]; p += 4 * m
    k = struct.unpack_from('<I', raw, p)[0]; p += 4 + 4 * k
    nm = struct.unpack_from('<H', raw, p)[0]; p += 2
    if nm == 0 or nm > 255:
        raise ValueError('materials')
    mats = []
    for _ in range(nm):
        _fl, sl = struct.unpack_from('<HH', raw, p); p += 4
        name = raw[p:p + sl].decode('cp949', 'replace'); p += sl
        if not name.lower().endswith(('.jpg', '.jpeg', '.tga', '.bmp', '.png')):
            raise ValueError('material name')
        _z, ni = struct.unpack_from('<II', raw, p); p += 8
        mats.append((name, list(struct.unpack_from('<%dH' % ni, raw, p)))); p += 2 * ni
    X = struct.unpack_from('<H', raw, p)[0]; p += 2
    if X % 3 or not 1 <= X // 3 <= 63:
        raise ValueError('bones')
    nb = X // 3
    bones = []
    for _ in range(nb):
        x, y, z, par = struct.unpack_from('<4h', raw, p); p += 8
        bones.append(((x, y, z), par))
    owner = [0] * nv
    for i in range(nb):
        cnt = struct.unpack_from('<H', raw, p)[0]; p += 2
        for v in struct.unpack_from('<%dH' % cnt, raw, p):
            if v < nv:
                owner[v] = i
        p += 2 * cnt
    return dict(local=local, weights=weights, uvs=uvs, materials=mats, bones=bones, owner=owner)


def parse_dcz(raw):
    if raw[0] == 0x81:
        return _parse_dcz_at(raw, 1)
    last = None
    for p in range(1, min(4096, len(raw) - 8)):
        try:
            return _parse_dcz_at(raw, p)
        except Exception as e:  # newer files start with a metadata block
            last = e
    raise ValueError('unsupported .dcz layout (%s)' % last)


def load_dcz(path):
    with open(path, 'rb') as f:
        d = parse_dcz(decode(f.read()))
    # rest positions of bones and vertices (bones have no rest rotation)
    world = []
    for (off, par) in d['bones']:
        base = world[par] if 0 <= par < len(world) else (0, 0, 0)
        world.append((base[0] + off[0], base[1] + off[1], base[2] + off[2]))
    d['bone_world'] = world
    d['verts'] = [tuple(world[o][k] + l[k] for k in range(3)) for o, l in zip(d['owner'], d['local'])]
    return d


def build_dcz(bone_world, parents, verts, owner, weights, uvs, materials):
    """bone_world: rest head of each of the 19 classic bones (mm); verts: rest positions (mm);
    owner: bone index per vertex; weights: 0..127 share of the owner (rest -> parent);
    uvs: (u, v) with v = 0 at the bottom; materials: [(texture file name, triangle indices)]"""
    nv = len(verts)
    if nv > 65535:
        raise ValueError('too many vertices (%d, max 65535) - decimate the mesh' % nv)
    out = bytearray(struct.pack('<BI', 0x81, nv * 3))
    clamp = lambda c: max(-32768, min(32767, int(round(c))))
    for v, o in zip(verts, owner):
        b = bone_world[o]
        out += struct.pack('<3h', clamp(v[0] - b[0]), clamp(v[1] - b[1]), clamp(v[2] - b[2]))
    out += struct.pack('<I', nv) + bytes(max(0, min(127, int(w))) for w in weights)
    out += struct.pack('<I', nv * 2)
    for u, v in uvs:
        out += struct.pack('<2f', float(u), float(v))
    out += struct.pack('<I', 0)
    out += struct.pack('<H', len(materials))
    for name, idx in materials:
        nb = name.encode('cp949', 'replace')
        out += struct.pack('<HH', dz.MAT_FLAGS, len(nb)) + nb + struct.pack('<II', 0, len(idx))
        out += struct.pack('<%dH' % len(idx), *idx)
    out += struct.pack('<H', 3 * len(bone_world))
    for i, b in enumerate(bone_world):
        p = parents[i]
        pb = bone_world[p] if p >= 0 else (0, 0, 0)
        out += struct.pack('<4h', clamp(b[0] - pb[0]), clamp(b[1] - pb[1]), clamp(b[2] - pb[2]), p)
    for i in range(len(bone_world)):
        vs = [v for v in range(nv) if owner[v] == i]
        out += struct.pack('<H', len(vs)) + (struct.pack('<%dH' % len(vs), *vs) if vs else b'')
    out += dz.adjacency(verts, materials)
    out += dz.TAIL
    return bytes(out)


def save_dcz(path, *a, **kw):
    with open(path, 'wb') as f:
        f.write(encode(build_dcz(*a, **kw)))


# ---------------------------------------------------------------- .dmp
def parse_dmp(data):
    if data[:2] != b'MP':
        raise ValueError('not a motion pool')
    pid = struct.unpack_from('<I', data, 3)[0]
    n = struct.unpack_from('<H', data, 7)[0]
    sizes = struct.unpack_from('<%dI' % n, data, 9)
    p = 9 + 4 * n
    motions = []
    for size in sizes:
        cs = struct.unpack_from('<I', data, p + 4)[0]
        raw = decode(data[p + 8:p + 8 + cs])
        segs, frames = raw[0], struct.unpack_from('<H', raw, 1)[0]
        root = [struct.unpack_from('<3h', raw, 3 + 6 * f) for f in range(frames)]
        q0 = 3 + 6 * frames
        nb = min(19, (len(raw) - q0) // (frames * 8))
        quats = []
        for f in range(frames):
            fr = []
            for b in range(19):
                bi = min(b, nb - 1)
                x, y, z, w = struct.unpack_from('<4h', raw, q0 + ((f * nb + bi) * 4) * 2)
                fr.append(q_norm((x / 10000.0, y / 10000.0, z / 10000.0, w / 10000.0)))
            quats.append(fr)
        motions.append(dict(segs=max(1, segs), frames=frames, root=root, quats=quats))
        p += size
    return dict(id=pid, motions=motions)


def build_dmp(pool_id, motions):
    """motions: [dict(segs, root=[(x,y,z) root units], quats=[[19 x (x,y,z,w)] per frame])]"""
    recs = []
    for i, m in enumerate(motions):
        frames = len(m['root'])
        if frames > 65535:
            raise ValueError('dance too long (max 65535 frames)')
        raw = bytearray(struct.pack('<BH', max(1, min(255, m['segs'])), frames))
        for r in m['root']:
            raw += struct.pack('<3h', *[max(-32768, min(32767, int(round(c)))) for c in r])
        for fr in m['quats']:
            for q in fr:
                q = q_norm(q)
                if q[3] < 0:
                    q = tuple(-c for c in q)
                raw += struct.pack('<4h', *[max(-10000, min(10000, int(round(c * 10000)))) for c in q])
        body = encode(bytes(raw))
        recs.append(struct.pack('<H', i) + b'MM' + struct.pack('<I', len(body)) + body)
    out = bytearray(b'MP\x01' + struct.pack('<IH', pool_id, len(recs)))
    for r in recs:
        out += struct.pack('<I', len(r))
    for r in recs:
        out += r
    return bytes(out)


def new_pool_id():
    return 0x77000000 | random.randint(0x100000, 0xFFFFFF)


# ---------------------------------------------------------------- .dsq
def parse_dsq(data):
    s = decode(data)
    out = dict(sound=None, slots=[], pools=[], pool_motion=[])
    i = s.find(b'SF')
    if 0 <= i < 64:
        ln = s[i + 2]
        out['sound'] = s[i + 3:i + 3 + ln].decode('cp949', 'replace')
    beats = None
    for j in range(len(s) - 8):
        if s[j:j + 2] == b'BS' and s[j + 2] == 1:
            q = j + 3 + 2
            n = struct.unpack_from('<H', s, q)[0]; q += 2
            beats = []
            for _ in range(n):
                c = s[q]; q += 1
                beats.append(sum(struct.unpack_from('<%dH' % c, s, q)) if c else 0); q += 2 * c
            break
    for j in range(len(s) - 8):
        if s[j] == 0x4d and s[j + 2] == 1 and s[j + 1] in b'S0U':
            p = j + 3
            n = struct.unpack_from('<H', s, p)[0]; p += 2
            dancers = s[p]; p += 1
            if not (0 < n < 4000 and 0 < dancers < 17):
                continue
            pool = struct.unpack_from('<%dH' % n, s, p); p += 2 * n
            seg = struct.unpack_from('<%dH' % n, s, p); p += 2 * n
            np_ = struct.unpack_from('<H', s, p)[0]; p += 2
            six = np_ > 1 and struct.unpack_from('<I', s, p)[0] == struct.unpack_from('<I', s, p + 6)[0]
            st = 6 if six else 4
            for k in range(np_):
                out['pools'].append(struct.unpack_from('<I', s, p + k * st)[0])
                out['pool_motion'].append(struct.unpack_from('<H', s, p + k * st + 4)[0] if six else -1)
            fb = next((b for b in (beats or []) if b), 2000)
            for k in range(n):
                ms = beats[k] if beats and k < len(beats) and beats[k] else fb
                out['slots'].append(dict(pool=pool[k], seg=seg[k], ms=ms))
            break
    if not out['slots']:
        raise ValueError('no motion block (locked iDance routine?)')
    out['cam'] = parse_cam_block(s, len(out['slots']))
    return out


# ---------------------------------------------------------------- camera work
# .dsq "CS" block: n slots; pool[n], seg[n], span[n] (u16, 0xffff = no camera move);
#                  u16 np, np x (u32 camera pool id, u16 motion index)
# .dcp camera pool: "CP" u8 1, u32 id, u16 n, n x u32 record size; record = u16 index, "CM", u32 size, zlib data
#                  data = u8 keys, keys x 7 f32: t (0..1 through the motion), panX, panY, z (= -distance mm),
#                  pitch, yaw, roll (degrees), orbiting a point 0.9 m above the dancer's feet.
CAM_PIVOT_MM = 900.0


def parse_cam_block(s, n):
    for j in range(len(s) - 8):
        if s[j:j + 3] != b'CS\x01':
            continue
        p = j + 3
        if struct.unpack_from('<H', s, p)[0] != n:
            continue
        p += 2
        if p + 6 * n + 2 > len(s):
            continue
        pool = struct.unpack_from('<%dH' % n, s, p)
        seg = struct.unpack_from('<%dH' % n, s, p + 2 * n)
        span = struct.unpack_from('<%dH' % n, s, p + 4 * n)
        p += 6 * n
        np_ = struct.unpack_from('<H', s, p)[0]; p += 2
        if np_ > 256 or p + 6 * np_ > len(s):
            continue
        pools = [(struct.unpack_from('<I', s, p + 6 * k)[0], struct.unpack_from('<H', s, p + 6 * k + 4)[0]) for k in range(np_)]
        if not pools:
            return None
        return dict(slots=[dict(pool=pool[k], seg=seg[k], span=span[k]) for k in range(n)], pools=pools)
    return None


def parse_dcp(data):
    if data[:2] != b'CP':
        raise ValueError('not a camera pool')
    pid = struct.unpack_from('<I', data, 3)[0]
    n = struct.unpack_from('<H', data, 7)[0]
    p = 9 + 4 * n
    motions = []
    for i in range(n):
        size = struct.unpack_from('<I', data, 9 + 4 * i)[0]
        keys = []
        try:
            cs = struct.unpack_from('<I', data, p + 4)[0]
            raw = decode(data[p + 8:p + 8 + cs])
            for k in range(raw[0]):
                if 1 + 28 * k + 28 <= len(raw):
                    keys.append(struct.unpack_from('<7f', raw, 1 + 28 * k))
        except Exception:
            pass
        motions.append(keys)
        p += size
    return dict(id=pid, motions=motions)


def build_dcp(pool_id, motions):
    """motions: [[(t, panX, panY, z, pitch, yaw, roll), ...] (2..255 keys each)]"""
    recs = []
    for i, keys in enumerate(motions):
        keys = keys[:255]
        raw = bytearray([len(keys)])
        for k in keys:
            raw += struct.pack('<7f', *k)
        body = zlib.compress(bytes(raw), 9)
        recs.append(struct.pack('<H', i) + b'CM' + struct.pack('<I', len(body)) + body)
    out = bytearray(b'CP\x01' + struct.pack('<IH', pool_id, len(recs)))
    for r in recs:
        out += struct.pack('<I', len(r))
    for r in recs:
        out += r
    return bytes(out)


def new_cam_pool_id():
    return 0x78000000 | random.randint(0x100000, 0xFFFFFF)


def find_cam_pool_files(dplayer_dir):
    """camera pool id -> .dcp path for everything under Sequences/"""
    out = {}
    for base, _dirs, files in os.walk(os.path.join(dplayer_dir, 'Sequences')):
        for f in files:
            if f.lower().endswith('.dcp'):
                p = os.path.join(base, f)
                try:
                    with open(p, 'rb') as fh:
                        h = fh.read(7)
                    if h[:2] == b'CP':
                        out.setdefault(struct.unpack_from('<I', h, 3)[0], p)
                except OSError:
                    pass
    return out


def sample_camera(dsq, cam_pools, fps=30):
    """Per frame (30 fps) camera values (panX, panY, z, pitch, yaw, roll) or None where the routine has no camera move."""
    cam = dsq.get('cam')
    if not cam:
        return None
    total = sum(sl['ms'] for sl in dsq['slots']) / 1000.0
    n = int(math.ceil(total * fps)) + 1
    out = [None] * n
    t0 = 0.0
    any_ = False
    for i, sl in enumerate(dsq['slots']):
        dur = sl['ms'] / 1000.0
        cs = cam['slots'][i] if i < len(cam['slots']) else None
        if cs and cs['pool'] != 0xFFFF and cs['pool'] < len(cam['pools']):
            pid, mi = cam['pools'][cs['pool']]
            pool = cam_pools.get(pid)
            keys = pool['motions'][mi] if pool and mi < len(pool['motions']) else None
            if keys:
                span = max(1, cs['span'] or 1)
                a, b = cs['seg'] / span, (cs['seg'] + 1) / span
                for k in range(int(math.floor(t0 * fps)), min(n - 1, int(math.ceil((t0 + dur) * fps))) + 1):
                    u = min(1.0, max(0.0, (k / fps - t0) / dur)) if dur > 0 else 0.0
                    t = a + (b - a) * u
                    j = 0
                    while j < len(keys) - 2 and keys[j + 1][0] < t:
                        j += 1
                    A, B_ = keys[j], keys[min(len(keys) - 1, j + 1)]
                    w = min(1.0, max(0.0, (t - A[0]) / (B_[0] - A[0]))) if B_[0] > A[0] else 0.0
                    v = [A[c] + (B_[c] - A[c]) * w for c in range(1, 7)]
                    dy = B_[5] - A[5]
                    if dy > 180: dy -= 360
                    elif dy < -180: dy += 360
                    v[4] = A[5] + dy * w
                    out[k] = tuple(v)
                    any_ = True
        t0 += dur
    return out if any_ else None


def build_dsq(pool_id, n_slots, slot_ms, sound=None, cam_pool_id=None):
    """One motion played straight through: slot i shows segment i for slot_ms[i] milliseconds.
    With cam_pool_id, slot i also plays camera motion i of that camera pool."""
    n = n_slots
    out = bytearray(b'SQ\x01' + struct.pack('<H', n))
    if sound:
        sb = sound.encode('cp949', 'replace')[:250]
        out += b'SF' + bytes([len(sb)]) + sb
    out += b'MS\x01' + struct.pack('<HB', n, 1)
    out += struct.pack('<%dH' % n, *([0] * n)) + struct.pack('<%dH' % n, *range(n))
    out += struct.pack('<H', 1) + struct.pack('<IH', pool_id, 0)
    if cam_pool_id is None:
        out += b'CS\x01' + struct.pack('<H', n) + b'\xff\xff' * (3 * n) + struct.pack('<H', 0)
    else:
        out += b'CS\x01' + struct.pack('<H', n) + struct.pack('<%dH' % n, *range(n)) + b'\x00\x00' * n + b'\x01\x00' * n
        out += struct.pack('<H', n) + b''.join(struct.pack('<IH', cam_pool_id, i) for i in range(n))
    out += b'BS\x01' + struct.pack('<HH', 0, n)
    for ms in slot_ms:
        q = [int(ms // 4)] * 4
        q[-1] = int(ms - q[0] * 3)
        out += bytes([4]) + struct.pack('<4H', *[max(0, min(65535, x)) for x in q])
    out += b'SE'
    return zlib.compress(bytes(out), 9)


def sample_routine(dsq, pools):
    """Frames at 30 fps: list of (root_mm(x,y,z), [19 local quats]) - how D-Player plays the routine."""
    pieces = []
    for sl in dsq['slots']:
        pid = dsq['pools'][sl['pool']] if sl['pool'] < len(dsq['pools']) else None
        pool = pools.get(pid)
        if not pool:
            continue
        mi = dsq['pool_motion'][sl['pool']] if sl['pool'] < len(dsq['pool_motion']) else -1
        g = sl['seg']
        if mi is not None and mi >= 0 and mi < len(pool['motions']):
            m = pool['motions'][mi]
            g = min(g, m['segs'] - 1)
        else:
            m = None
            for mm in pool['motions']:
                if g < mm['segs']:
                    m = mm
                    break
                g -= mm['segs']
            if m is None:
                continue
        seg_len = m['frames'] / m['segs']
        pieces.append((m, g * seg_len, min(m['frames'] - 1, (g + 1) * seg_len), sl['ms'] / 1000.0))
    total = sum(p[3] for p in pieces)
    frames = []
    n = max(2, int(math.ceil(total * FPS)) + 1)
    pi, acc = 0, 0.0
    sc = HIPS_MM / ROOT_REF
    for k in range(n):
        t = min(total, k / FPS)
        while pi < len(pieces) - 1 and t > acc + pieces[pi][3]:
            acc += pieces[pi][3]
            pi += 1
        m, f0, f1, dur = pieces[pi]
        u = min(1.0, (t - acc) / dur) if dur > 0 else 0.0
        f = min(m['frames'] - 1, max(0.0, f0 + u * (f1 - f0)))
        fa = int(f)
        fb = min(m['frames'] - 1, fa + 1)
        w = f - fa
        qs = [q_slerp(m['quats'][fa][b], m['quats'][fb][b], w) for b in range(19)]
        r = tuple((m['root'][fa][i] * (1 - w) + m['root'][fb][i] * w) * sc for i in range(3))
        frames.append((r, qs))
    return frames


def find_pool_files(dplayer_dir):
    """pool id -> .dmp path for everything under Sequences/"""
    out = {}
    seq = os.path.join(dplayer_dir, 'Sequences')
    for base, _dirs, files in os.walk(seq):
        for f in files:
            if f.lower().endswith('.dmp'):
                p = os.path.join(base, f)
                try:
                    with open(p, 'rb') as fh:
                        h = fh.read(7)
                    if h[:2] == b'MP':
                        out.setdefault(struct.unpack_from('<I', h, 3)[0], p)
                except OSError:
                    pass
    return out
