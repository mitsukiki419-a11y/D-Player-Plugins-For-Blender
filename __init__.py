bl_info = {
    "name": "D-Player Stage Tools",
    "author": "Claude for Mathew Isip",
    "version": (1, 0, 0),
    "blender": (2, 93, 0),
    "location": "File > Import/Export, and 3D View > Sidebar (N) > D-Player",
    "description": "Import and export D-Player 2.0 stages (.dsz) and export stages for D-Player FX (.glb)",
    "category": "Import-Export",
}

import os
import re
import shutil

import bpy
import bmesh
from bpy.props import StringProperty, FloatProperty, BoolProperty, PointerProperty, EnumProperty
from bpy_extras.io_utils import ImportHelper, ExportHelper
from mathutils import Vector

from . import dsz_format

# D-Player: millimetres, +Y up, character faces +Z (toward the default camera).
# Blender:  metres,      +Z up, front view looks along +Y (so "toward camera" = -Y).
MM = 1000.0


def bl_to_dp(co, scale):
    return (co.x * MM * scale, co.z * MM * scale, -co.y * MM * scale)


def dp_to_bl(v, scale):
    return Vector((v[0] / MM / scale, -v[2] / MM / scale, v[1] / MM / scale))


def safe_name(s):
    s = re.sub(r'[^A-Za-z0-9_\-]+', '_', s).strip('_')
    return s or 'stage'


# --------------------------------------------------------------------------- settings
class DPlayerSettings(bpy.types.PropertyGroup):
    dplayer_dir: StringProperty(
        name="D-Player folder",
        description="Your D-Player install folder (the one with D-Player20.exe, Stages, Characters ...)",
        subtype='DIR_PATH',
        default=r"C:\Users\Mathew Isip\Documents\D-Player",
    )
    stage_name: StringProperty(name="Stage name", default="MyStage")
    selected_only: BoolProperty(name="Selected objects only", default=False)
    scale: FloatProperty(
        name="Scale", default=1.0, min=0.01, max=100.0,
        description="Extra scale applied on export. 1.0 = 1 Blender metre is 1 metre in D-Player",
    )
    max_texture: EnumProperty(
        name="Max texture size",
        items=[('256', '256', ''), ('512', '512', ''), ('1024', '1024', ''), ('2048', '2048', '')],
        default='1024',
        description="Textures are resized to a power of two no bigger than this (old D-Player likes power-of-two textures)",
    )


# --------------------------------------------------------------------------- helpers
def material_image(mat):
    """Return the image used for a material's colour, or None."""
    if mat is None or not mat.use_nodes or not mat.node_tree:
        return None
    nodes = mat.node_tree.nodes
    out = None
    for n in nodes:
        if n.type == 'BSDF_PRINCIPLED':
            inp = n.inputs.get('Base Color')
            if inp and inp.is_linked:
                src = inp.links[0].from_node
                # walk back through simple nodes (mix, hue/sat, etc.)
                seen = 0
                while src and src.type != 'TEX_IMAGE' and seen < 6:
                    nxt = None
                    for i in src.inputs:
                        if i.is_linked:
                            nxt = i.links[0].from_node
                            break
                    src = nxt
                    seen += 1
                if src is not None and src.type == 'TEX_IMAGE' and src.image:
                    return src.image
    for n in nodes:
        if n.type == 'TEX_IMAGE' and n.image:
            out = n.image
            break
    return out


def material_color(mat):
    if mat is None:
        return (0.8, 0.8, 0.8, 1.0)
    if mat.use_nodes and mat.node_tree:
        for n in mat.node_tree.nodes:
            if n.type == 'BSDF_PRINCIPLED':
                c = n.inputs['Base Color'].default_value
                return (c[0], c[1], c[2], 1.0)
    c = mat.diffuse_color
    return (c[0], c[1], c[2], 1.0)


def material_uses_alpha(mat):
    if mat is None:
        return False
    m = getattr(mat, 'blend_method', 'OPAQUE')
    if m in ('BLEND', 'CLIP', 'HASHED'):
        return True
    return getattr(mat, 'surface_render_method', 'DITHERED') == 'BLENDED'


def pot(n, cap):
    p = 1
    while p < n and p < cap:
        p *= 2
    return min(p, cap)


def write_texture(mat, image, tex_dir, cap, used_names):
    """Save a power-of-two copy of the material's texture (or a colour swatch). Returns file name."""
    alpha = material_uses_alpha(mat)
    ext = '.tga' if alpha else '.jpg'
    if image is not None:
        base = safe_name(os.path.splitext(image.name)[0])
    else:
        base = safe_name((mat.name if mat else 'default') + '_color')
    name = base + ext
    i = 1
    while name.lower() in used_names:
        name = '%s_%d%s' % (base, i, ext)
        i += 1
    used_names.add(name.lower())
    path = os.path.join(tex_dir, name)

    if image is not None and image.size[0] > 0:
        img = image.copy()
        w, h = img.size
        img.scale(pot(w, cap), pot(h, cap))
    else:
        img = bpy.data.images.new('dp_swatch', 8, 8, alpha=False)
        col = material_color(mat)
        img.pixels[:] = list(col) * 64
    img.filepath_raw = path
    img.file_format = 'TARGA' if alpha else 'JPEG'
    try:
        img.save()
    except RuntimeError:
        # fall back to save_render (handles generated / float images)
        scn = bpy.context.scene
        settings = scn.render.image_settings
        old = settings.file_format
        settings.file_format = 'TARGA' if alpha else 'JPEG'
        img.save_render(path, scene=scn)
        settings.file_format = old
    bpy.data.images.remove(img)
    return name


def write_thumbnail(stage_dir, name, first_tex):
    """64x64 BMP preview D-Player shows in its stage list."""
    path = os.path.join(stage_dir, name + '.bmp')
    img = None
    if first_tex and os.path.exists(first_tex):
        try:
            img = bpy.data.images.load(first_tex, check_existing=False)
            img.scale(64, 64)
        except Exception:
            img = None
    if img is None:
        img = bpy.data.images.new('dp_thumb', 64, 64)
        img.pixels[:] = [0.2, 0.3, 0.6, 1.0] * 64 * 64
    img.filepath_raw = path
    img.file_format = 'BMP'
    img.save()
    bpy.data.images.remove(img)


def collect_meshes(context, selected_only):
    objs = context.selected_objects if selected_only else context.scene.objects
    out = []
    for o in objs:
        if o.type in {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT'} and o.visible_get():
            out.append(o)
    return out


def gather_geometry(context, objects, scale):
    """Triangulate evaluated meshes; return verts, uvs, {material_key: (mat, [indices])}."""
    depsgraph = context.evaluated_depsgraph_get()
    verts, uvs = [], []
    vmap = {}
    groups = {}
    order = []
    for obj in objects:
        ev = obj.evaluated_get(depsgraph)
        try:
            mesh = ev.to_mesh()
        except RuntimeError:
            continue
        if mesh is None:
            continue
        mesh.calc_loop_triangles()
        mw = ev.matrix_world
        uv_layer = mesh.uv_layers.active.data if mesh.uv_layers.active else None
        mats = list(ev.material_slots)
        flip = mw.determinant() < 0
        for tri in mesh.loop_triangles:
            mat = mats[tri.material_index].material if tri.material_index < len(mats) else None
            key = mat.name if mat else '__default__'
            if key not in groups:
                groups[key] = (mat, [])
                order.append(key)
            idx_list = groups[key][1]
            loops = list(tri.loops)
            if flip:
                loops = [loops[0], loops[2], loops[1]]
            for li in loops:
                vi = mesh.loops[li].vertex_index
                co = bl_to_dp(mw @ mesh.vertices[vi].co, scale)
                uv = tuple(uv_layer[li].uv) if uv_layer else (0.0, 0.0)
                k = (round(co[0]), round(co[1]), round(co[2]), round(uv[0], 5), round(uv[1], 5))
                if k not in vmap:
                    vmap[k] = len(verts)
                    verts.append(co)
                    uvs.append(uv)
                idx_list.append(vmap[k])
        ev.to_mesh_clear()
    return verts, uvs, [(k, groups[k][0], groups[k][1]) for k in order]


# --------------------------------------------------------------------------- export .dsz
def export_dsz(context, stage_dir, name, selected_only, scale, cap, report):
    objects = collect_meshes(context, selected_only)
    if not objects:
        report({'ERROR'}, "Nothing to export - no visible mesh objects")
        return False
    verts, uvs, groups = gather_geometry(context, objects, scale)
    if not verts:
        report({'ERROR'}, "The meshes have no faces")
        return False
    if len(verts) > 65535:
        report({'ERROR'}, "Too many vertices (%d). D-Player allows 65,535 - decimate or simplify the stage" % len(verts))
        return False
    big = max(max(abs(c) for c in v) for v in verts)
    if big > 32767:
        report({'WARNING'}, "Stage is bigger than D-Player allows (about 32 m from the centre); parts were clamped. Lower the Scale")

    os.makedirs(stage_dir, exist_ok=True)
    tex_folder = name + '_tex'
    tex_dir = os.path.join(stage_dir, tex_folder)
    os.makedirs(tex_dir, exist_ok=True)
    used = set()
    materials = []
    first_tex = None
    for key, mat, idx in groups:
        if len(idx) // 3 > 65535:
            report({'ERROR'}, "Material %s has too many triangles; split it into two materials" % key)
            return False
        fname = write_texture(mat, material_image(mat), tex_dir, cap, used)
        if first_tex is None:
            first_tex = os.path.join(tex_dir, fname)
        materials.append(('./%s/%s' % (tex_folder, fname), idx))
    if len(materials) > 255:
        report({'ERROR'}, "Too many materials (max 255)")
        return False
    dsz_format.save(os.path.join(stage_dir, name + '.dsz'), verts, uvs, materials)
    write_thumbnail(stage_dir, name, first_tex)
    tris = sum(len(i) for _, i in materials) // 3
    report({'INFO'}, "Exported %s: %d triangles, %d materials -> %s" % (name, tris, len(materials), stage_dir))
    return True


class DPLAYER_OT_export_dsz(bpy.types.Operator, ExportHelper):
    """Export the scene as a D-Player 2.0 stage (.dsz + textures)"""
    bl_idname = "export_scene.dplayer_dsz"
    bl_label = "Export D-Player Stage (.dsz)"
    filename_ext = ".dsz"
    filter_glob: StringProperty(default="*.dsz", options={'HIDDEN'})
    selected_only: BoolProperty(name="Selected only", default=False)
    scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100)
    max_texture: EnumProperty(name="Max texture size",
                              items=[('256', '256', ''), ('512', '512', ''), ('1024', '1024', ''), ('2048', '2048', '')],
                              default='1024')

    def execute(self, context):
        name = safe_name(os.path.splitext(os.path.basename(self.filepath))[0])
        stage_dir = os.path.dirname(self.filepath)
        ok = export_dsz(context, stage_dir, name, self.selected_only, self.scale, int(self.max_texture), self.report)
        return {'FINISHED'} if ok else {'CANCELLED'}


# --------------------------------------------------------------------------- import .dsz
def import_dsz(context, path, scale=1.0):
    stage = dsz_format.load(path)
    base_dir = os.path.dirname(path)
    name = os.path.splitext(os.path.basename(path))[0]
    verts = stage['verts']
    uvs = stage['uvs']

    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bverts = [bm.verts.new(dp_to_bl(v, scale)) for v in verts]
    uv_layer = bm.loops.layers.uv.new('UVMap')
    mat_index = {}
    for mi, (tex, idx) in enumerate(stage['materials']):
        mat_index[mi] = len(mesh.materials)
        mesh.materials.append(make_material(tex, base_dir))
        for t in range(len(idx) // 3):
            a, b, c = idx[3 * t:3 * t + 3]
            if len({a, b, c}) < 3:
                continue
            try:
                f = bm.faces.new((bverts[a], bverts[b], bverts[c]))
            except ValueError:
                # duplicate face in the same vertices: make a copy so nothing is lost
                nv = [bm.verts.new(bverts[i].co) for i in (a, b, c)]
                f = bm.faces.new(nv)
            f.material_index = mat_index[mi]
            for loop, vi in zip(f.loops, (a, b, c)):
                loop[uv_layer].uv = uvs[vi]
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    context.collection.objects.link(obj)
    for o in context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj
    return obj


def find_file_ci(base_dir, rel):
    rel = rel.replace('\\', '/')
    if rel.startswith('./'):
        rel = rel[2:]
    p = os.path.join(base_dir, rel)
    if os.path.exists(p):
        return p
    cur = base_dir
    for part in rel.split('/'):
        if not os.path.isdir(cur):
            return None
        hit = None
        for e in os.listdir(cur):
            if e.lower() == part.lower():
                hit = e
                break
        if hit is None:
            return None
        cur = os.path.join(cur, hit)
    return cur


def make_material(tex_rel, base_dir):
    mname = os.path.splitext(os.path.basename(tex_rel.replace('\\', '/')))[0]
    mat = bpy.data.materials.new(mname)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get('Principled BSDF')
    path = find_file_ci(base_dir, tex_rel)
    if bsdf is not None:
        if 'Specular' in bsdf.inputs:
            bsdf.inputs['Specular'].default_value = 0.0
        elif 'Specular IOR Level' in bsdf.inputs:
            bsdf.inputs['Specular IOR Level'].default_value = 0.0
        bsdf.inputs['Roughness'].default_value = 1.0
    if path and bsdf is not None:
        tex = nt.nodes.new('ShaderNodeTexImage')
        tex.location = (-350, 250)
        try:
            tex.image = bpy.data.images.load(path, check_existing=True)
        except RuntimeError:
            return mat
        nt.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
        if path.lower().endswith('.tga'):
            nt.links.new(tex.outputs['Alpha'], bsdf.inputs['Alpha'])
            if hasattr(mat, 'blend_method'):
                try:
                    mat.blend_method = 'BLEND'
                except Exception:
                    pass
    return mat


class DPLAYER_OT_import_dsz(bpy.types.Operator, ImportHelper):
    """Import a D-Player stage (.dsz) so you can edit it"""
    bl_idname = "import_scene.dplayer_dsz"
    bl_label = "Import D-Player Stage (.dsz)"
    filename_ext = ".dsz"
    filter_glob: StringProperty(default="*.dsz", options={'HIDDEN'})

    def execute(self, context):
        try:
            obj = import_dsz(context, self.filepath)
        except Exception as e:
            self.report({'ERROR'}, "Could not read stage: %s" % e)
            return {'CANCELLED'}
        s = context.scene.dplayer
        s.stage_name = safe_name(obj.name) + "_edit"
        self.report({'INFO'}, "Imported %s" % obj.name)
        return {'FINISHED'}


# --------------------------------------------------------------------------- panel operators
def stage_root(settings):
    root = bpy.path.abspath(settings.dplayer_dir)
    return root


class DPLAYER_OT_quick_export(bpy.types.Operator):
    """Export straight into <D-Player>/Stages/<Stage name>/ so it shows up in D-Player"""
    bl_idname = "dplayer.quick_export_dsz"
    bl_label = "Export to D-Player (.dsz)"

    def execute(self, context):
        s = context.scene.dplayer
        root = stage_root(s)
        if not os.path.isdir(root):
            self.report({'ERROR'}, "Set your D-Player folder first")
            return {'CANCELLED'}
        name = safe_name(s.stage_name)
        stage_dir = os.path.join(root, 'Stages', name)
        ok = export_dsz(context, stage_dir, name, s.selected_only, s.scale, int(s.max_texture), self.report)
        return {'FINISHED'} if ok else {'CANCELLED'}


class DPLAYER_OT_export_fx(bpy.types.Operator):
    """Export to <D-Player>/Stages3D/<Stage name>/<Stage name>.glb for D-Player FX (keeps full quality, lights and alpha)"""
    bl_idname = "dplayer.export_fx_stage"
    bl_label = "Export to D-Player FX (.glb)"

    def execute(self, context):
        s = context.scene.dplayer
        root = stage_root(s)
        if not os.path.isdir(root):
            self.report({'ERROR'}, "Set your D-Player folder first")
            return {'CANCELLED'}
        name = safe_name(s.stage_name)
        out_dir = os.path.join(root, 'Stages3D', name)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name + '.glb')
        kw = dict(filepath=path, export_format='GLB', use_selection=s.selected_only, export_apply=True,
                  export_lights=True, export_cameras=False, export_yup=True)
        try:
            bpy.ops.export_scene.gltf(**kw)
        except TypeError:
            kw.pop('export_lights', None)
            bpy.ops.export_scene.gltf(**kw)
        self.report({'INFO'}, "Exported %s" % path)
        return {'FINISHED'}


class DPLAYER_OT_import_from_dplayer(bpy.types.Operator):
    """Pick one of the stages in your D-Player Stages folder to edit"""
    bl_idname = "dplayer.import_from_dplayer"
    bl_label = "Open D-Player Stage"
    bl_property = "stage"

    def stage_items(self, context):
        root = stage_root(context.scene.dplayer)
        items = []
        sdir = os.path.join(root, 'Stages')
        if os.path.isdir(sdir):
            for d in sorted(os.listdir(sdir), key=str.lower):
                p = os.path.join(sdir, d)
                if os.path.isdir(p):
                    for f in os.listdir(p):
                        if f.lower().endswith('.dsz'):
                            items.append((os.path.join(p, f), d, ''))
                            break
        return items or [('NONE', 'No stages found', '')]

    stage: EnumProperty(items=stage_items)

    def invoke(self, context, event):
        context.window_manager.invoke_search_popup(self)
        return {'FINISHED'}

    def execute(self, context):
        if self.stage == 'NONE':
            return {'CANCELLED'}
        try:
            obj = import_dsz(context, self.stage)
        except Exception as e:
            self.report({'ERROR'}, "Could not read stage: %s" % e)
            return {'CANCELLED'}
        context.scene.dplayer.stage_name = safe_name(obj.name) + "_edit"
        return {'FINISHED'}


class DPLAYER_OT_add_scale_ref(bpy.types.Operator):
    """Add a 1.7 m tall dancer-sized box at the stage centre (where the D-Player character stands, facing -Y / the front view)"""
    bl_idname = "dplayer.add_scale_ref"
    bl_label = "Add dancer size guide"

    def execute(self, context):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.85))
        o = context.active_object
        o.scale = (0.5, 0.3, 1.7)
        o.name = "DancerGuide (not exported)"
        o.display_type = 'WIRE'
        o.hide_render = True
        o.hide_set(False)
        # keep it out of exports
        o.hide_select = False
        o["dplayer_ignore"] = True
        return {'FINISHED'}


_orig_collect = collect_meshes


def collect_meshes(context, selected_only):  # noqa: F811 - skip guide objects
    return [o for o in _orig_collect(context, selected_only) if not o.get("dplayer_ignore")]


class DPLAYER_PT_panel(bpy.types.Panel):
    bl_label = "D-Player"
    bl_idname = "DPLAYER_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "D-Player"

    def draw(self, context):
        s = context.scene.dplayer
        col = self.layout.column()
        col.prop(s, "dplayer_dir")
        col.separator()
        col.operator("dplayer.import_from_dplayer", icon='FILE_FOLDER')
        col.operator("dplayer.add_scale_ref", icon='OUTLINER_OB_ARMATURE')
        col.separator()
        col.prop(s, "stage_name")
        col.prop(s, "selected_only")
        col.prop(s, "scale")
        col.prop(s, "max_texture")
        col.separator()
        col.operator("dplayer.quick_export_dsz", icon='EXPORT')
        col.operator("dplayer.export_fx_stage", icon='EXPORT')
        box = col.box()
        box.label(text="Floor = Z 0, dancer at the origin")
        box.label(text="Dancer faces the front view (-Y)")
        box.label(text="Classic D-Player: max ~32 m wide")


def menu_import(self, context):
    self.layout.operator(DPLAYER_OT_import_dsz.bl_idname, text="D-Player Stage (.dsz)")


def menu_export(self, context):
    self.layout.operator(DPLAYER_OT_export_dsz.bl_idname, text="D-Player Stage (.dsz)")


classes = (
    DPlayerSettings,
    DPLAYER_OT_export_dsz,
    DPLAYER_OT_import_dsz,
    DPLAYER_OT_quick_export,
    DPLAYER_OT_export_fx,
    DPLAYER_OT_import_from_dplayer,
    DPLAYER_OT_add_scale_ref,
    DPLAYER_PT_panel,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.dplayer = PointerProperty(type=DPlayerSettings)
    bpy.types.TOPBAR_MT_file_import.append(menu_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_export)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_export)
    del bpy.types.Scene.dplayer
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
