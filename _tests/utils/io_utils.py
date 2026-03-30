"""USD geometry I/O utilities for interpolation test scripts."""

import numpy as np
from pxr import Usd, UsdGeom, Vt

import shutil
from pathlib import Path
from pxr import Sdf

def load_usd_file(stage_path):
    """
    Load a USD file.

    Args:
        stage_path (str): Path to the USD file

    Returns:
        Usd.Stage: USD stage
    """
    stage = Usd.Stage.Open(stage_path)
    if not stage:
        raise ValueError(f"Failed to open USD stage at: {stage_path}")
    return stage


def get_usd_mesh_vertices(usd_stage, mesh_path):
    """
    Get the vertices of a USD mesh.

    Args:
        usd_stage (Usd.Stage): USD stage
        mesh_path (str): Path to the USD mesh (e.g., '/World/Mesh')

    Returns:
        np.array: Vertex positions (N x 3)
    """
    prim = usd_stage.GetPrimAtPath(mesh_path)
    if not prim.IsValid():
        raise ValueError(f"No valid prim found at path: {mesh_path}")

    mesh = UsdGeom.Mesh(prim)
    if not mesh:
        raise ValueError(f"Prim at {mesh_path} is not a mesh")

    points_attr = mesh.GetPointsAttr()
    points = points_attr.Get()

    vertices = np.array(points, dtype=np.float32)
    return vertices


def update_usd_mesh_vertices(usd_stage, mesh_path, new_vertices):
    """
    Update the vertices of a USD mesh.

    Args:
        usd_stage (Usd.Stage): USD stage
        mesh_path (str): Path to the USD mesh (e.g., '/World/Mesh')
        new_vertices (np.array): New vertex positions (N x 3)

    Returns:
        Usd.Stage: USD stage with updated mesh
    """
    prim = usd_stage.GetPrimAtPath(mesh_path)
    if not prim.IsValid():
        raise ValueError(f"No valid prim found at path: {mesh_path}")

    mesh = UsdGeom.Mesh(prim)
    if not mesh:
        raise ValueError(f"Prim at {mesh_path} is not a mesh")

    if new_vertices.dtype != np.float32:
        new_vertices = new_vertices.astype(np.float32)

    points_array = Vt.Vec3fArray.FromBuffer(new_vertices.flatten())

    points_attr = mesh.GetPointsAttr()
    points_attr.Set(points_array)

    return usd_stage


def get_usd_mesh_topology(usd_stage, mesh_path):
    """
    Get the full topology and subdivision attributes of a USD mesh.

    Args:
        usd_stage (Usd.Stage): USD stage
        mesh_path (str): Path to the USD mesh (e.g., '/geometry/geometry')

    Returns:
        dict: Mesh topology data with keys:
            - 'points': np.array (N, 3) float32
            - 'face_vertex_counts': np.array (F,) int
            - 'face_vertex_indices': np.array (I,) int
            - 'subdivision_scheme': str ('catmullClark', 'loop', 'bilinear', 'none')
            - 'interpolate_boundary': str ('none', 'edgeOnly', 'edgeAndCorner')
            - 'crease_indices': np.array int or None
            - 'crease_lengths': np.array int or None
            - 'crease_sharpnesses': np.array float or None
            - 'corner_indices': np.array int or None
            - 'corner_sharpnesses': np.array float or None
    """
    prim = usd_stage.GetPrimAtPath(mesh_path)
    if not prim.IsValid():
        raise ValueError(f"No valid prim found at path: {mesh_path}")

    mesh = UsdGeom.Mesh(prim)
    if not mesh:
        raise ValueError(f"Prim at {mesh_path} is not a mesh")

    points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32)
    face_vertex_counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    face_vertex_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)

    scheme = mesh.GetSubdivisionSchemeAttr().Get() or 'none'
    interp_boundary = mesh.GetInterpolateBoundaryAttr().Get() or 'edgeAndCorner'

    def _get_optional_attr(attr):
        val = attr.Get()
        if val is not None and len(val) > 0:
            return np.array(val)
        return None

    return {
        'points': points,
        'face_vertex_counts': face_vertex_counts,
        'face_vertex_indices': face_vertex_indices,
        'subdivision_scheme': str(scheme),
        'interpolate_boundary': str(interp_boundary),
        'crease_indices': _get_optional_attr(mesh.GetCreaseIndicesAttr()),
        'crease_lengths': _get_optional_attr(mesh.GetCreaseLengthsAttr()),
        'crease_sharpnesses': _get_optional_attr(mesh.GetCreaseSharpnessesAttr()),
        'corner_indices': _get_optional_attr(mesh.GetCornerIndicesAttr()),
        'corner_sharpnesses': _get_optional_attr(mesh.GetCornerSharpnessesAttr()),
    }


def create_usd_mesh(usd_stage, mesh_path, points, face_vertex_counts, face_vertex_indices,
                     subdivision_scheme='none', interpolate_boundary='edgeAndCorner',
                     crease_indices=None, crease_lengths=None, crease_sharpnesses=None,
                     corner_indices=None, corner_sharpnesses=None):
    """
    Create a new USD mesh with full topology and optional subdivision attributes.

    Args:
        usd_stage (Usd.Stage): USD stage
        mesh_path (str): Prim path for the new mesh
        points (np.array): Vertex positions (N, 3)
        face_vertex_counts (np.array or list): Vertices per face
        face_vertex_indices (np.array or list): Flat vertex index array
        subdivision_scheme (str): 'catmullClark', 'bilinear', 'loop', or 'none'
        interpolate_boundary (str): 'none', 'edgeOnly', or 'edgeAndCorner'
        crease_indices (np.array, optional): Crease edge vertex indices
        crease_lengths (np.array, optional): Number of vertices per crease
        crease_sharpnesses (np.array, optional): Sharpness per crease
        corner_indices (np.array, optional): Corner vertex indices
        corner_sharpnesses (np.array, optional): Sharpness per corner

    Returns:
        UsdGeom.Mesh: The created mesh
    """
    from pxr import Gf

    mesh = UsdGeom.Mesh.Define(usd_stage, mesh_path)

    pts = np.asarray(points, dtype=np.float32)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([int(c) for c in face_vertex_counts]))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([int(i) for i in face_vertex_indices]))

    mesh.GetSubdivisionSchemeAttr().Set(subdivision_scheme)
    mesh.GetInterpolateBoundaryAttr().Set(interpolate_boundary)

    if crease_indices is not None and crease_lengths is not None and crease_sharpnesses is not None:
        mesh.GetCreaseIndicesAttr().Set(Vt.IntArray([int(i) for i in crease_indices]))
        mesh.GetCreaseLengthsAttr().Set(Vt.IntArray([int(l) for l in crease_lengths]))
        mesh.GetCreaseSharpnessesAttr().Set(Vt.FloatArray([float(s) for s in crease_sharpnesses]))

    if corner_indices is not None and corner_sharpnesses is not None:
        mesh.GetCornerIndicesAttr().Set(Vt.IntArray([int(i) for i in corner_indices]))
        mesh.GetCornerSharpnessesAttr().Set(Vt.FloatArray([float(s) for s in corner_sharpnesses]))

    return mesh


def export_usd_with_textures(usd_stage, output_path):
    """
    Export a USD stage and copy all referenced texture files to the output directory.
    Updates material paths to reference the copied textures.

    Args:
        usd_stage (Usd.Stage): USD stage to export
        output_path (str): Path where to save the USD file
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_path)

    texture_attributes = []
    stage_real_path = usd_stage.GetRootLayer().realPath or ""
    stage_root_dir = Path(stage_real_path).parent if stage_real_path else Path.cwd()

    for prim in usd_stage.Traverse():
        for attr in prim.GetAttributes():
            try:
                attr_value = attr.Get()
                attr_str = str(attr_value) if attr_value else ""
                if any(attr_str.lower().endswith(ext) for ext in ['.exr', '.png', '.jpg', '.jpeg', '.tiff', '.tx']):
                    texture_attributes.append((prim, attr.GetName(), attr_value))
            except Exception:
                pass

    for prim, attr_name, original_path in texture_attributes:
        attr = prim.GetAttribute(attr_name)
        if not attr:
            continue

        orig_str = str(original_path)
        texture_path = Path(orig_str)
        if texture_path.exists():
            source_texture = texture_path
        else:
            relative_source = stage_root_dir / orig_str
            if relative_source.exists():
                source_texture = relative_source
            else:
                continue

        dest_texture = output_dir / source_texture.name
        if source_texture.exists():
            shutil.copy2(str(source_texture), str(dest_texture))
            attr.Set(Sdf.AssetPath(source_texture.name))

    usd_stage.Export(output_path)


def add_lines_to_usd_stage(usd_stage, lines, line_path='/World/Lines', color=(1.0, 0.0, 0.0), line_width=2.0):
    """
    Add lines to a USD stage as BasisCurves.

    Args:
        usd_stage (Usd.Stage): USD stage
        lines (np.array): Array of line segments shape (N, 2, 3) where each line has 2 points
        line_path (str): Path where to add the lines in the stage
        color (tuple): RGB color for the lines (values 0-1)
        line_width (float): Width of the lines (default 2.0)

    Returns:
        Usd.Stage: USD stage with added lines
    """
    from pxr import Gf

    usd_stage.DefinePrim(line_path, 'Scope')

    all_points = []
    curve_vertex_counts = []

    for line in lines:
        all_points.extend([tuple(line[0]), tuple(line[1])])
        curve_vertex_counts.append(2)

    curves_path = f"{line_path}/Curves"
    curves_prim = usd_stage.DefinePrim(curves_path, 'BasisCurves')
    curves = UsdGeom.BasisCurves(curves_prim)

    points_array = Vt.Vec3fArray([Gf.Vec3f(*p) for p in all_points])
    curves.GetPointsAttr().Set(points_array)

    curves.GetCurveVertexCountsAttr().Set(Vt.IntArray(curve_vertex_counts))

    curves.GetTypeAttr().Set('linear')
    curves.GetBasisAttr().Set('bezier')

    color_attr = curves.GetDisplayColorPrimvar()
    if not color_attr:
        color_attr = curves.CreateDisplayColorPrimvar()
    color_attr.Set([Gf.Vec3f(*color)])

    width_attr = curves.GetWidthsAttr()
    widths = [line_width] * len(all_points)
    width_attr.Set(Vt.FloatArray(widths))

    return usd_stage
