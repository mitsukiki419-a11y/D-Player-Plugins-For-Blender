"""Blender operators for classic D-Player characters (.dcz) and dances (.dsq / .dmp)."""
import math
import os
import re
import shutil

import bpy
import bmesh
from bpy.props import StringProperty, BoolProperty
from bpy_extras.io_utils import ImportHelper, ExportHelper
from mathutils import Vector, Quaternion, Matrix

from . import classic_format as cf

MM = 1000.0
B = cf.CLASSIC_BONES
PAR = cf.PARENTS
# reference skeleton (mm) used when a dance is imported with no armature selected
REF_OFFSETS = [[0, 1100, 0], [0, 70, 0], [0, 150, 0], [0, 260, 0], [0, 50, 0], [70, 220, -10], [80, -20, 0], [240, 0, 0], [250, 0, 0],
               [-70, 220, -10], [-80, -20, 0], [-240, 0, 0], [-250, 0, 0], [80, -80, 0], [10, -400, 0], [20, -400, 0], [-80, -80, 0], [-10, -400, 0], [-20, -400, 0]]


# D-Player (mm, +Y up, faces +Z)  <->  Blender (m, +Z up, faces -Y)
def dp2bl(v):
    return Vector((v[0] / MM, -v[2] / MM, v[1] / MM))


def bl2dp(v):
    return (v.x * MM, v.z * MM, -v.y * MM)


def qdp2bl(q):          # q = (x, y, z, w) in D-Player space -> mathutils Quaternion (w, x, y, z)
    return Quaternion((q[3], q[0], -q[2], q[1]))


def qbl2dp(q):
    return (q.x, q.z, -q.y, q.w)


def settings(context):
    return context.scene.dplayer


# ---------------------------------------------------------------- bone names
SIDE_RE = re.compile(r'^(?:left|l)[_\.\- ]?(.*)$|^(.*?)[_\.\- ](?:l|left)$', re.I)


def canon(name):
    """Map a bone name from Mixamo / Unreal / Blender rigs / VRoid / D-Player to a classic bone name (or None)."""
    n = name.split(':')[-1].split('|')[-1]
    n = re.sub(r'^(mixamorig\d*|bip0?1|j_bip_[clr]_|def[-_]|orig[-_])', '', n, flags=re.I)
    n = re.sub(r'[_\.\- ]?(jnt|joint|bone)$', '', n, flags=re.I)
    side = ''
    m = re.match(r'^([LR])[_\.\- ](.+)$', n) or re.match(r'^(left|right)[_\.\- ]?(.+)$', n, re.I)
    if m:
        side = 'Left' if m.group(1).lower().startswith('l') else 'Right'
        n = m.group(2)
    else:
        m = re.match(r'^(.+?)[_\.\- ]?([LR]|left|right)$', n, re.I)
        if m and (len(m.group(2)) > 1 or re.search(r'[_\.\- ]', n)):
            side = 'Left' if m.group(2).lower().startswith('l') else 'Right'
            n = m.group(1)
    k = re.sub(r'[^a-z0-9]', '', n.lower())
    if not side:
        if k in ('hips', 'pelvis', 'hip', 'cog'):
            return 'Hips'
        if k in ('spine', 'spine0', 'spine00', 'spine01', 'abdomen', 'spinelower'):
            return 'Spine'
        if k in ('spine2', 'spine02', 'spine03', 'chest', 'upperchest', 'spineupper'):
            return 'Spine2'
        if k in ('spine1',):
            return 'Spine1'
        if k in ('neck', 'neck01', 'neck1'):
            return 'Neck'
        if k in ('head',):
            return 'Head'
        return None
    table = {
        'Shoulder': ('shoulder', 'clavicle', 'collar'),
        'Arm': ('arm', 'upperarm', 'uparm'),
        'ForeArm': ('forearm', 'lowerarm', 'elbow'),
        'Hand': ('hand', 'wrist'),
        'UpLeg': ('upleg', 'thigh', 'upperleg'),
        'Leg': ('leg', 'calf', 'shin', 'lowerleg', 'knee'),
        'Foot': ('foot', 'ankle'),
    }
    for c, keys in table.items():
        if k in keys:
            return side + c
    return None


def map_armature(arm):
    """classic name -> bone name in this armature (Spine2 falls back to Spine1/chest when needed)"""
    found = {}
    spines = []
    for b in arm.data.bones:
        c = canon(b.name)
        if c and c not in found:
            found[c] = b.name
        if re.search(r'spine|chest', b.name, re.I):
            spines.append(b.name)
    if 'Spine' not in found and spines:
        found['Spine'] = spines[0]
    s1 = found.pop('Spine1', None)
    if 'Spine2' not in found and s1:
        found['Spine2'] = s1
    if 'Spine2' not in found and len(spines) > 1:
        found['Spine2'] = spines[-1]
    return found


# ---------------------------------------------------------------- textures
def resolve_texture(char_dir, name):
    rel = name.replace('\\', '/').lstrip('./')
    p = os.path.join(char_dir, *rel.split('/'))
    if os.path.exists(p):
        return p
    stem = os.path.splitext(os.path.basename(rel))[0].lower()
    files = []
    for base, _d, fs in os.walk(char_dir):
        for f in fs:
            files.append(os.path.join(base, f))
    # <stem>.txt texture swap file: first <map> entry
    for f in files:
        if os.path.basename(f).lower() == stem + '.txt':
            try:
                txt = open(f, 'rb').read().decode('cp949', 'replace')
                m = re.search(r'<\s*map\s*>\s*(?:\d+\s+)?([^\r\n<]+)', txt, re.I)
                if m:
                    q = resolve_texture(char_dir, m.group(1).strip()) if m.group(1).strip().lower() != rel.lower() else None
                    if q:
                        return q
            except OSError:
                pass
    for f in files:
        b = os.path.basename(f).lower()
        if os.path.splitext(b)[0] == stem and b.endswith(('.jpg', '.jpeg', '.bmp', '.tga', '.png')):
            return f
    return None


def make_material(name, tex_path):
    mat = bpy.data.materials.new(os.path.splitext(os.path.basename(name))[0])
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if tex_path and bsdf:
        tn = nt.nodes.new('ShaderNodeTexImage')
        try:
            tn.image = bpy.data.images.load(tex_path, check_existing=True)
        except RuntimeError:
            tn.image = None
        nt.links.new(tn.outputs['Color'], bsdf.inputs['Base Color'])
        if tex_path.lower().endswith('.tga'):
            nt.links.new(tn.outputs['Alpha'], bsdf.inputs['Alpha'])
            mat.blend_method = 'CLIP'
        if 'Roughness' in bsdf.inputs:
            bsdf.inputs['Roughness'].default_value = 0.8
    mat['dplayer_texture'] = name
    return mat


# ---------------------------------------------------------------- armature
def build_armature(context, name, bone_world_mm):
    arm_data = bpy.data.armatures.new(name)
    arm = bpy.data.objects.new(name, arm_data)
    context.collection.objects.link(arm)
    context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    heads = [dp2bl(p) for p in bone_world_mm]
    kids = {i: [j for j in range(19) if PAR[j] == i] for i in range(19)}
    ebs = []
    for i, n in enumerate(B):
        eb = arm_data.edit_bones.new(n)
        eb.head = heads[i]
        if kids[i]:
            # point at the main child (spine for the hips, the next limb bone otherwise)
            c = kids[i][0]
            tail = heads[c].copy()
        elif PAR[i] >= 0:
            d = heads[i] - heads[PAR[i]]
            tail = heads[i] + (d.normalized() * 0.08 if d.length > 1e-4 else Vector((0, 0, 0.08)))
        else:
            tail = heads[i] + Vector((0, 0, 0.1))
        if n == 'Head':
            tail = heads[i] + Vector((0, 0, 0.2))
        if (tail - eb.head).length < 0.01:
            tail = eb.head + Vector((0, 0, 0.05))
        eb.tail = tail
        eb.roll = 0
        ebs.append(eb)
    for i, eb in enumerate(ebs):
        if PAR[i] >= 0:
            eb.parent = ebs[PAR[i]]
    bpy.ops.object.mode_set(mode='OBJECT')
    arm.show_in_front = True
    return arm


# ---------------------------------------------------------------- import character
def import_dcz(context, path):
    d = cf.load_dcz(path)
    name = os.path.splitext(os.path.basename(path))[0]
    char_dir = os.path.dirname(path)
    nb = len(d['bones'])
    if nb != 19:
        raise ValueError('This character has %d bones; only the standard 19-bone D-Player skeleton is supported' % nb)
    # stand on the floor (D-Player characters are stored with their own origin)
    low = min(v[1] for v in d['verts'])
    d['verts'] = [(v[0], v[1] - low, v[2]) for v in d['verts']]
    d['bone_world'] = [(b[0], b[1] - low, b[2]) for b in d['bone_world']]
    arm = build_armature(context, name + '_rig', d['bone_world'])
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bverts = [bm.verts.new(dp2bl(v)) for v in d['verts']]
    bm.verts.index_update()
    uvl = bm.loops.layers.uv.new('UVMap')
    extra = {}                                   # duplicated vertices (same triangle used twice) -> original
    for mi, (n, idx) in enumerate(d['materials']):
        me.materials.append(make_material(n, resolve_texture(char_dir, n)))
        for t in range(len(idx) // 3):
            tri = idx[3 * t:3 * t + 3]
            if len(set(tri)) < 3:
                continue
            try:
                f = bm.faces.new([bverts[i] for i in tri])
            except ValueError:
                nv = []
                for i in tri:
                    v = bm.verts.new(bverts[i].co)
                    nv.append(v)
                    extra[v] = i
                f = bm.faces.new(nv)
            f.material_index = mi
            for loop, vi in zip(f.loops, tri):
                loop[uvl].uv = d['uvs'][vi]
    bm.verts.index_update()
    dup = {v.index: i for v, i in extra.items()}
    bm.to_mesh(me)
    bm.free()
    me.update()
    ob = bpy.data.objects.new(name, me)
    context.collection.objects.link(ob)
    groups = [ob.vertex_groups.new(name=n) for n in B]
    src = list(range(len(d['verts']))) + [None] * (len(me.vertices) - len(d['verts']))
    for k, i in dup.items():
        src[k] = i
    for vi in range(len(me.vertices)):
        o, w = d['owner'][src[vi]], d['weights'][src[vi]]
        share = max(0, min(127, w)) / 127.0
        groups[o].add([vi], share if PAR[o] >= 0 else 1.0, 'REPLACE')
        if PAR[o] >= 0 and share < 1.0:
            groups[PAR[o]].add([vi], 1.0 - share, 'ADD')
    ob.parent = arm
    mod = ob.modifiers.new('Armature', 'ARMATURE')
    mod.object = arm
    ob['dplayer_character'] = name
    return arm, ob


# ---------------------------------------------------------------- dances
def classic_armature(context):
    ob = context.active_object
    if ob and ob.type == 'MESH' and ob.parent and ob.parent.type == 'ARMATURE':
        ob = ob.parent
    if ob and ob.type == 'ARMATURE':
        return ob
    # no armature selected: make the reference D-Player skeleton
    world = []
    for i, off in enumerate(REF_OFFSETS):
        base = world[PAR[i]] if PAR[i] >= 0 else (0, 0, 0)
        world.append((base[0] + off[0], base[1] + off[1], base[2] + off[2]))
    return build_armature(context, 'DPlayer_rig', world)


def apply_frames(context, arm, frames, action_name):
    """frames: [(root_mm, [19 local quats (D-Player)])] at 30 fps -> Blender action on arm."""
    mapping = map_armature(arm)
    if 'Hips' not in mapping:
        raise ValueError('The selected armature has no hips/pelvis bone')
    scn = context.scene
    scn.render.fps = 30
    scn.render.fps_base = 1.0
    act = bpy.data.actions.new(action_name)
    arm.animation_data_create()
    arm.animation_data.action = act
    bones = arm.data.bones
    rest_q = {c: bones[n].matrix_local.to_quaternion() for c, n in mapping.items()}
    hips_rest = bones[mapping['Hips']].head_local.copy()
    hips_scale = max(0.2, hips_rest.z) / (cf.HIPS_MM / MM)        # dances are made for a 1.1 m hips height
    n = len(frames)
    for c, bn in mapping.items():
        pb = arm.pose.bones[bn]
        pb.rotation_mode = 'QUATERNION'
        bi = B.index(c)
        qr = rest_q[c]
        qri = qr.inverted()
        vals = [[], [], [], []]
        prev = None
        for _r, qs in frames:
            # D-Player local rotation (rest = no rotation) -> pose basis of this Blender bone
            # (missing in-between classic bones are folded into their child)
            q = qdp2bl(qs[bi])
            p = PAR[bi]
            while p >= 0 and B[p] not in mapping:
                q = qdp2bl(qs[p]) @ q
                p = PAR[p]
            basis = qri @ q @ qr
            if prev is not None and prev.dot(basis) < 0:
                basis.negate()
            prev = basis
            for k in range(4):
                vals[k].append(basis[k])
        path = 'pose.bones["%s"].rotation_quaternion' % bn
        for k in range(4):
            fc = act.fcurves.new(path, index=k, action_group=bn)
            fc.keyframe_points.add(n)
            co = []
            for f in range(n):
                co += [f + 1, vals[k][f]]
            fc.keyframe_points.foreach_set('co', co)
            fc.update()
    # hips travel
    hb = mapping['Hips']
    qri = rest_q['Hips'].inverted()
    vals = [[], [], []]
    for r, _qs in frames:
        pos = dp2bl(r) * hips_scale
        delta = qri @ (pos - hips_rest)
        for k in range(3):
            vals[k].append(delta[k])
    for k in range(3):
        fc = act.fcurves.new('pose.bones["%s"].location' % hb, index=k, action_group=hb)
        fc.keyframe_points.add(n)
        co = []
        for f in range(n):
            co += [f + 1, vals[k][f]]
        fc.keyframe_points.foreach_set('co', co)
        fc.update()
    scn.frame_start = 1
    scn.frame_end = n
    return act


def add_song(context, song_path):
    if not song_path or not os.path.exists(song_path):
        return False
    scn = context.scene
    if not scn.sequence_editor:
        scn.sequence_editor_create()
    for s in list(scn.sequence_editor.sequences):
        if s.type == 'SOUND' and s.name.startswith('DP_'):
            scn.sequence_editor.sequences.remove(s)
    st = scn.sequence_editor.sequences.new_sound('DP_' + os.path.basename(song_path), song_path, 1, 1)
    st.show_waveform = True
    return True


def find_song(dplayer_dir, name):
    if not name:
        return None
    stem = os.path.splitext(os.path.basename(name.replace('\\', '/')))[0].lower()
    for sub in ('Sounds', 'Music'):
        d = os.path.join(dplayer_dir, sub)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            fs, ext = os.path.splitext(f)
            if fs.lower() == stem and ext.lower() in ('.mp3', '.wav', '.ogg'):
                return os.path.join(d, f)
    return None


def guess_dplayer_dir(path, fallback):
    p = os.path.dirname(path)
    for _ in range(4):
        if os.path.isdir(os.path.join(p, 'Sequences')) and os.path.isdir(os.path.join(p, 'Characters')):
            return p
        p = os.path.dirname(p)
    return fallback


# ---------------------------------------------------------------- camera work
# D-Player cameras orbit a pivot 0.9 m above the dancer's feet: yaw/pitch turn the orbit, z is minus the distance,
# panX/panY slide camera and target sideways/up in the camera's own frame.
CAM_FOV = 40.0          # vertical field of view used by D-Player FX
_C = Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))           # D-Player axes -> Blender axes


def _rot_dp(pitch, yaw, roll):
    return (Matrix.Rotation(math.radians(yaw), 3, 'Y') @ Matrix.Rotation(math.radians(-pitch), 3, 'X')
            @ Matrix.Rotation(math.radians(roll), 3, 'Z'))


def dp_cam_matrix(v):
    """(panX, panY, z, pitch, yaw, roll) -> Blender camera world matrix"""
    px, py, z, pitch, yaw, roll = v
    R = _rot_dp(pitch, yaw, roll)
    pos = Vector((0, cf.CAM_PIVOT_MM, 0)) + R @ Vector((px, py, abs(z)))
    m = (_C @ R).to_4x4()
    m.translation = dp2bl(pos)
    return m


def cam_dp_values(mw):
    """Blender camera world matrix -> (panX, panY, z, pitch, yaw, roll)"""
    R = _C.transposed() @ mw.to_3x3().normalized()
    f = -R.col[2]
    yaw = math.degrees(math.atan2(-f.x, -f.z))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -f.y))))
    R0 = _rot_dp(pitch, yaw, 0)
    Rz = R0.transposed() @ R
    roll = math.degrees(math.atan2(Rz[1][0], Rz[0][0]))
    local = _rot_dp(pitch, yaw, roll).transposed() @ (Vector(bl2dp(mw.translation)) - Vector((0, cf.CAM_PIVOT_MM, 0)))
    d = max(100.0, local.z)
    return (local.x, local.y, -d, pitch, yaw, roll)


def dplayer_camera(context, name='DP_Camera'):
    ob = bpy.data.objects.get(name)
    if ob is None or ob.type != 'CAMERA':
        cd = bpy.data.cameras.new(name)
        ob = bpy.data.objects.new(name, cd)
        context.scene.collection.objects.link(ob)
    cd = ob.data
    cd.sensor_fit = 'VERTICAL'
    cd.angle_y = math.radians(CAM_FOV)
    cd.clip_start, cd.clip_end = 0.05, 500
    context.scene.camera = ob
    return ob


def apply_camera(context, values, name):
    """values: per 30 fps frame (panX, panY, z, pitch, yaw, roll) or None -> keyframed DP_Camera"""
    ob = dplayer_camera(context)
    ob.rotation_mode = 'XYZ'
    act = bpy.data.actions.new(name + ' camera')
    ob.animation_data_create()
    ob.animation_data.action = act
    locs, rots = [], []
    prev = None
    last = None
    for v in values:
        if v is None:
            v = last
        if v is None:
            v = (0, 0, -3000, 5, 0, 0)
        last = v
        m = dp_cam_matrix(v)
        e = m.to_euler('XYZ', prev) if prev else m.to_euler('XYZ')
        prev = e
        locs.append(m.translation.copy())
        rots.append(e)
    n = len(values)
    for path, data in (('location', locs), ('rotation_euler', rots)):
        for k in range(3):
            fc = act.fcurves.new(path, index=k)
            fc.keyframe_points.add(n)
            co = []
            for f in range(n):
                co += [f + 1, data[f][k]]
            fc.keyframe_points.foreach_set('co', co)
            for kp in fc.keyframe_points:
                kp.interpolation = 'LINEAR'
            fc.update()
    return ob


def import_dance(context, path):
    ext = os.path.splitext(path)[1].lower()
    name = os.path.splitext(os.path.basename(path))[0]
    dp_dir = guess_dplayer_dir(path, settings(context).dplayer_dir)
    song = None
    cam_vals = None
    if ext == '.dmp':
        pool = cf.parse_dmp(open(path, 'rb').read())
        sc = cf.HIPS_MM / cf.ROOT_REF
        frames = []
        for m in pool['motions']:
            for f in range(m['frames']):
                frames.append((tuple(c * sc for c in m['root'][f]), m['quats'][f]))
    else:
        dsq = cf.parse_dsq(open(path, 'rb').read())
        files = cf.find_pool_files(dp_dir)
        pools = {}
        for pid in dsq['pools']:
            if pid in files:
                pools[pid] = cf.parse_dmp(open(files[pid], 'rb').read())
        if not pools:
            raise ValueError('The motion file for this routine is not in Sequences/Motion Pool')
        frames = cf.sample_routine(dsq, pools)
        song = find_song(dp_dir, dsq['sound'])
        if dsq.get('cam'):
            cfiles = cf.find_cam_pool_files(dp_dir)
            cpools = {}
            for pid, _m in dsq['cam']['pools']:
                if pid in cfiles and pid not in cpools:
                    try:
                        cpools[pid] = cf.parse_dcp(open(cfiles[pid], 'rb').read())
                    except Exception:
                        pass
            cam_vals = cf.sample_camera(dsq, cpools)
    arm = classic_armature(context)
    apply_frames(context, arm, frames, name)
    if cam_vals:
        apply_camera(context, cam_vals, name)
        context.scene.frame_end = max(context.scene.frame_end, len(cam_vals))
    if song:
        add_song(context, song)
    else:
        se = context.scene.sequence_editor
        if se:
            for st in list(se.sequences):
                if st.type == 'SOUND' and st.name.startswith('DP_'):
                    se.sequences.remove(st)
    return arm, len(frames), song, bool(cam_vals)


def export_dance(context, arm, dp_dir, name, song_path=None, camera=None):
    """Sample the armature's animation at 30 fps and write Sequences/<name>.dsq + Motion Pool/<name>.dmp"""
    mapping = map_armature(arm)
    missing = [c for c in ('Hips', 'LeftArm', 'RightArm', 'LeftUpLeg', 'RightUpLeg') if c not in mapping]
    if missing:
        raise ValueError('Could not find these bones in the armature: ' + ', '.join(missing))
    scn = context.scene
    f0, f1 = scn.frame_start, scn.frame_end
    step = (scn.render.fps / scn.render.fps_base) / 30.0     # sample at 30 fps whatever the scene rate
    mw = arm.matrix_world
    mq = mw.to_quaternion()
    bones = arm.data.bones
    rest_w = {c: mq @ bones[n].matrix_local.to_quaternion() for c, n in mapping.items()}
    hips_rest_h = (mw @ bones[mapping['Hips']].head_local).z
    if hips_rest_h <= 1e-4:
        hips_rest_h = 1.0
    root_scale = cf.ROOT_REF / (hips_rest_h * MM)
    roots, quats, cams = [], [], []
    t = float(f0)
    old = scn.frame_current
    while t <= f1 + 1e-6:
        scn.frame_set(int(t), subframe=t - int(t))
        D = [None] * 19
        for i, c in enumerate(B):
            if c in mapping:
                pb = arm.pose.bones[mapping[c]]
                Dw = (mq @ pb.matrix.to_quaternion()) @ rest_w[c].inverted()    # world-space change from rest
                D[i] = Dw
            else:
                D[i] = D[PAR[i]] if PAR[i] >= 0 else Quaternion()
        fr = []
        for i in range(19):
            L = (D[PAR[i]].inverted() @ D[i]) if PAR[i] >= 0 else D[i]
            fr.append(qbl2dp(L.normalized()))
        quats.append(fr)
        head = mw @ arm.pose.bones[mapping['Hips']].head
        dpp = bl2dp(head)
        roots.append(tuple(c * root_scale for c in dpp))
        if camera is not None:
            cams.append(cam_dp_values(camera.matrix_world.copy()))
        t += step
    scn.frame_set(old)
    frames = len(roots)
    if frames < 2:
        raise ValueError('The animation is empty - set the scene frame range')
    seg_frames = 60                                   # one routine slot per 2 seconds
    segs = max(1, min(255, int(math.ceil(frames / seg_frames))))
    seg_ms = frames / segs / 30.0 * 1000.0
    pid = cf.new_pool_id()
    seq_dir = os.path.join(dp_dir, 'Sequences')
    pool_dir = os.path.join(seq_dir, 'Motion Pool')
    os.makedirs(pool_dir, exist_ok=True)
    with open(os.path.join(pool_dir, name + '.dmp'), 'wb') as f:
        f.write(cf.build_dmp(pid, [dict(segs=segs, root=roots, quats=quats)]))
    sound = None
    if song_path and os.path.exists(song_path):
        sdir = os.path.join(dp_dir, 'Sounds')
        os.makedirs(sdir, exist_ok=True)
        sound = os.path.basename(song_path)
        dst = os.path.join(sdir, sound)
        if os.path.abspath(dst) != os.path.abspath(song_path):
            shutil.copyfile(song_path, dst)
    cam_id = None
    if cams:
        # one camera move per routine part, sampled every 2 frames (D-Player keeps up to 255 keys per move)
        L = frames / segs
        motions = []
        for i in range(segs):
            K = max(2, min(255, int(math.ceil(L / 2)) + 1))
            keys = []
            prev_yaw = None
            for j in range(K):
                u = j / (K - 1)
                v = list(cams[min(frames - 1, int(round(i * L + u * L)))])
                if prev_yaw is not None:                      # keep yaw continuous inside a move
                    while v[4] - prev_yaw > 180: v[4] -= 360
                    while v[4] - prev_yaw < -180: v[4] += 360
                prev_yaw = v[4]
                keys.append((u, *v))
            motions.append(keys)
        cam_id = cf.new_cam_pool_id()
        cam_dir = os.path.join(seq_dir, 'Camera Pool')
        os.makedirs(cam_dir, exist_ok=True)
        with open(os.path.join(cam_dir, name + '.dcp'), 'wb') as f:
            f.write(cf.build_dcp(cam_id, motions))
    with open(os.path.join(seq_dir, name + '.dsq'), 'wb') as f:
        f.write(cf.build_dsq(pid, segs, [seg_ms] * segs, sound, cam_id))
    return frames, segs, sound, bool(cam_id)


def scene_song(context):
    se = context.scene.sequence_editor
    if not se:
        return None
    for s in se.sequences_all:
        if s.type == 'SOUND' and s.sound and s.sound.filepath:
            return bpy.path.abspath(s.sound.filepath)
    return None


# ---------------------------------------------------------------- export character
def export_character(context, mesh_objs, arm, dp_dir, name, cap, write_texture, material_image, write_thumbnail):
    mapping = map_armature(arm)
    if 'Hips' not in mapping:
        raise ValueError('The armature needs a hips/pelvis bone')
    mw = arm.matrix_world
    bones = arm.data.bones
    # each armature bone -> classic bone (its own, or the nearest mapped ancestor's)
    inv = {v: B.index(k) for k, v in mapping.items()}

    def classic_of(bname):
        b = bones.get(bname)
        while b is not None:
            if b.name in inv:
                return inv[b.name]
            b = b.parent
        return 0
    # rest positions of the 19 classic bones (mm)
    world = [None] * 19
    for i, c in enumerate(B):
        if c in mapping:
            world[i] = bl2dp(mw @ bones[mapping[c]].head_local)
    for i in range(19):
        if world[i] is None:
            world[i] = world[PAR[i]] if PAR[i] >= 0 else (0, 1000, 0)
    char_dir = os.path.join(dp_dir, 'Characters', name)
    tex_dir = os.path.join(char_dir, name + '_tex')
    os.makedirs(tex_dir, exist_ok=True)
    old_pose = arm.data.pose_position
    arm.data.pose_position = 'REST'
    context.view_layer.update()
    deps = context.evaluated_depsgraph_get()
    verts, uvs, owner, weights = [], [], [], []
    mats = {}          # texture file name -> index list
    used = set()
    tex_of_mat = {}
    first_tex = None
    try:
        for ob in mesh_objs:
            ev = ob.evaluated_get(deps)
            me = ev.to_mesh()
            bm = bmesh.new()
            bm.from_mesh(me)
            bmesh.ops.triangulate(bm, faces=bm.faces[:])
            uvl = bm.loops.layers.uv.active
            dvl = bm.verts.layers.deform.active
            groups = {g.index: g.name for g in ob.vertex_groups}
            omw = ob.matrix_world
            cache = {}
            for f in bm.faces:
                mat = ob.material_slots[f.material_index].material if f.material_index < len(ob.material_slots) else None
                key = mat.name if mat else '__none__'
                if key not in tex_of_mat:
                    tn = write_texture(mat, material_image(mat), tex_dir, cap, used)
                    tex_of_mat[key] = './' + name + '_tex/' + tn
                    if first_tex is None:
                        first_tex = os.path.join(tex_dir, tn)
                tname = tex_of_mat[key]
                lst = mats.setdefault(tname, [])
                tri = []
                for loop in f.loops:
                    uv = tuple(loop[uvl].uv) if uvl else (0.0, 0.0)
                    k = (ob.name, loop.vert.index, round(uv[0], 5), round(uv[1], 5))
                    if k not in cache:
                        v = loop.vert
                        cache[k] = len(verts)
                        verts.append(bl2dp(omw @ v.co))
                        uvs.append(uv)
                        share = {}
                        if dvl is not None:
                            for gi, w in v[dvl].items():
                                if gi in groups and w > 0:
                                    ci = classic_of(groups[gi])
                                    share[ci] = share.get(ci, 0) + w
                        if not share:
                            share = {0: 1.0}
                        o = max(share, key=share.get)
                        p = PAR[o]
                        wp = share.get(p, 0.0) if p >= 0 else 0.0
                        owner.append(o)
                        weights.append(int(round(127 * share[o] / (share[o] + wp))) if wp > 0 else 127)
                    tri.append(cache[k])
                lst += tri
            bm.free()
            ev.to_mesh_clear()
    finally:
        arm.data.pose_position = old_pose
    if len(verts) > 65535:
        raise ValueError('Too many vertices (%d). D-Player characters can have at most 65535 - decimate the mesh.' % len(verts))
    materials = [(n, idx) for n, idx in mats.items()]
    cf.save_dcz(os.path.join(char_dir, name + '.dcz'), world, PAR, verts, owner, weights, uvs, materials)
    try:
        write_thumbnail(char_dir, name, first_tex)
    except Exception:
        pass
    return len(verts), len(materials)


# ---------------------------------------------------------------- operators
class DPLAYER_OT_import_dcz(bpy.types.Operator, ImportHelper):
    """Import a classic D-Player character (.dcz) with its skeleton and textures"""
    bl_idname = "import_scene.dplayer_dcz"
    bl_label = "Import D-Player Character (.dcz)"
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = ".dcz"
    filter_glob: StringProperty(default="*.dcz", options={'HIDDEN'})

    def execute(self, context):
        try:
            if context.object and context.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            arm, ob = import_dcz(context, self.filepath)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self.report({'INFO'}, 'Imported %s (%d vertices)' % (ob.name, len(ob.data.vertices)))
        return {'FINISHED'}


class DPLAYER_OT_import_dance(bpy.types.Operator, ImportHelper):
    """Import a classic D-Player dance (.dsq routine or .dmp motion) onto the selected armature
(or a new D-Player skeleton); the routine's song is added to the timeline"""
    bl_idname = "import_anim.dplayer_dance"
    bl_label = "Import D-Player Dance (.dsq / .dmp)"
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = ".dsq"
    filter_glob: StringProperty(default="*.dsq;*.dmp", options={'HIDDEN'})

    def execute(self, context):
        try:
            if context.object and context.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            arm, n, song, has_cam = import_dance(context, self.filepath)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self.report({'INFO'}, 'Dance on %s: %d frames (%.1f s)%s%s' % (arm.name, n, n / 30.0, ', with song' if song else '', ', with camera work (DP_Camera)' if has_cam else ''))
        return {'FINISHED'}


class DPLAYER_OT_export_dance(bpy.types.Operator):
    """Save the active armature's animation as a D-Player dance (Sequences/<name>.dsq + Motion Pool/<name>.dmp).
Works with D-Player, Mixamo and most humanoid rigs whose rest pose is a T-pose"""
    bl_idname = "dplayer.export_dance"
    bl_label = "Export Dance to D-Player (.dsq)"

    def execute(self, context):
        s = settings(context)
        ob = context.active_object
        if ob and ob.type == 'MESH' and ob.parent and ob.parent.type == 'ARMATURE':
            ob = ob.parent
        if not ob or ob.type != 'ARMATURE':
            self.report({'ERROR'}, 'Select the dancing armature first')
            return {'CANCELLED'}
        if not os.path.isdir(bpy.path.abspath(s.dplayer_dir)):
            self.report({'ERROR'}, 'Set your D-Player folder in the D-Player panel')
            return {'CANCELLED'}
        name = re.sub(r'[\\/:*?"<>|]+', '_', s.dance_name.strip() or ob.name)
        song = bpy.path.abspath(s.dance_song) if s.dance_song else scene_song(context)
        try:
            cam = context.scene.camera if s.dance_camera else None
            frames, segs, sound, has_cam = export_dance(context, ob, bpy.path.abspath(s.dplayer_dir), name, song, cam)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self.report({'INFO'}, 'Saved %s.dsq: %.1f s in %d parts%s%s' % (name, frames / 30.0, segs, (', song ' + sound) if sound else '',
                                                                       ', camera work from ' + cam.name if has_cam else ''))
        return {'FINISHED'}


class DPLAYER_OT_export_character(bpy.types.Operator):
    """Save the selected skinned mesh(es) as a D-Player character (Characters/<name>/<name>.dcz).
The armature needs a T-pose rest pose; Mixamo and D-Player rigs work as they are"""
    bl_idname = "dplayer.export_character"
    bl_label = "Export Character to D-Player (.dcz)"

    def execute(self, context):
        from . import write_texture, material_image, write_thumbnail
        s = settings(context)
        meshes = [o for o in context.selected_objects if o.type == 'MESH']
        if not meshes and context.active_object and context.active_object.type == 'ARMATURE':
            arm = context.active_object
            meshes = [o for o in context.scene.objects if o.type == 'MESH' and any(m.type == 'ARMATURE' and m.object == arm for m in o.modifiers)]
        arm = None
        for o in meshes:
            for m in o.modifiers:
                if m.type == 'ARMATURE' and m.object:
                    arm = m.object
            if not arm and o.parent and o.parent.type == 'ARMATURE':
                arm = o.parent
        if not meshes or not arm:
            self.report({'ERROR'}, 'Select the character mesh(es) - they need an Armature modifier')
            return {'CANCELLED'}
        if not os.path.isdir(bpy.path.abspath(s.dplayer_dir)):
            self.report({'ERROR'}, 'Set your D-Player folder in the D-Player panel')
            return {'CANCELLED'}
        name = re.sub(r'[\\/:*?"<>|]+', '_', s.character_name.strip() or meshes[0].name)
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        try:
            nv, nm = export_character(context, meshes, arm, bpy.path.abspath(s.dplayer_dir), name, int(s.max_texture),
                                      write_texture, material_image, write_thumbnail)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self.report({'INFO'}, 'Saved Characters/%s/%s.dcz (%d vertices, %d textures)' % (name, name, nv, nm))
        return {'FINISHED'}


class DPLAYER_OT_open_character(bpy.types.Operator):
    """Pick a character from your D-Player Characters folder"""
    bl_idname = "dplayer.open_character"
    bl_label = "Open D-Player Character"

    def invoke(self, context, event):
        s = settings(context)
        d = os.path.join(bpy.path.abspath(s.dplayer_dir), 'Characters')
        return bpy.ops.import_scene.dplayer_dcz('INVOKE_DEFAULT', filepath=d + os.sep if os.path.isdir(d) else '')


class DPLAYER_OT_open_dance(bpy.types.Operator):
    """Pick a dance from your D-Player Sequences folder"""
    bl_idname = "dplayer.open_dance"
    bl_label = "Open D-Player Dance"

    def invoke(self, context, event):
        s = settings(context)
        d = os.path.join(bpy.path.abspath(s.dplayer_dir), 'Sequences')
        return bpy.ops.import_anim.dplayer_dance('INVOKE_DEFAULT', filepath=d + os.sep if os.path.isdir(d) else '')


class DPLAYER_OT_add_camera(bpy.types.Operator):
    """Add a camera set up like D-Player's (40° view, looking at the dancer from the front).
Animate it and it is saved with the dance"""
    bl_idname = "dplayer.add_camera"
    bl_label = "Add D-Player camera"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ob = dplayer_camera(context)
        ob.matrix_world = dp_cam_matrix((0, 0, -3500, 5, 0, 0))
        for o in context.selected_objects:
            o.select_set(False)
        ob.select_set(True)
        context.view_layer.objects.active = ob
        self.report({'INFO'}, 'DP_Camera is the scene camera - keyframe it (I key) to make camera moves')
        return {'FINISHED'}


classes = (
    DPLAYER_OT_add_camera,
    DPLAYER_OT_import_dcz,
    DPLAYER_OT_import_dance,
    DPLAYER_OT_export_dance,
    DPLAYER_OT_export_character,
    DPLAYER_OT_open_character,
    DPLAYER_OT_open_dance,
)
