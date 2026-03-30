#%%

# =========================================
# Import necessary libraries
# =========================================

import sys
from pathlib import Path

# Path setup that works both as script and in Jupyter
try:
    # When running as a script
    _tests_dir = Path(__file__).resolve().parent
    _parent_dir = str(_tests_dir.parent)
except NameError:
    # When running in Jupyter/interactive - __file__ doesn't exist
    _parent_dir = str(Path.cwd())
    _tests_dir = Path.cwd() / '_tests'

if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

_tests_root = _tests_dir
_submodule_root = Path(_parent_dir)

output_dir = Path(_tests_root) / "output"
output_dir.mkdir(parents=True, exist_ok=True)

import numpy as np

import utils.io_utils as io
from utils.usd_viewer import open_usd_viewer

from importlib import reload


# %%

# =========================================
# Loading the geometry
# =========================================

input_dir = Path(__file__).resolve().parent / "input"

quad_high_poly_mesh_path = str(input_dir / "Quad_TestVault_highPoly.usd")
quad_low_poly_mesh_path = str(input_dir / "Quad_TestVault_lowPoly.usd")

mesh_paths = [quad_high_poly_mesh_path, quad_low_poly_mesh_path]

mesh_path = '/geometry/geometry'

# Common control quads (same as test_usd_planar_interpolations.py)
base_quad = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
target_quad = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.7, 0.7], [0.5, 0.5]])


def make_fan_control_mesh(n_cols=2, n_rows=3, r_inner=0.3, r_outer=0.9, angle_deg=70):
    """
    Create a fan / sector control mesh (n_cols x n_rows quads).

    The shape is a circular sector: narrow at the inner radius, wide at the
    outer radius.  Left and right edges are straight radial lines; horizontal
    rows follow circular arcs.

    Vertex grid: (n_rows+1) rows × (n_cols+1) columns.
    Returns points (2D XZ), fvc, fvi, uv_ranges.
    """
    half_angle = np.radians(angle_deg / 2)
    radii = np.linspace(r_inner, r_outer, n_rows + 1)
    angles = np.linspace(-half_angle, half_angle, n_cols + 1)

    # Build vertices row-by-row (bottom = inner arc, top = outer arc)
    points = []
    for r in radii:
        for a in angles:
            x = r * np.sin(a)
            z = r * np.cos(a) - r_inner  # shift so inner arc ~= z=0
            points.append([x, z])
    points = np.array(points, dtype=np.float64)

    cols = n_cols + 1  # verts per row
    fvc_list, fvi_list, uv_ranges = [], [], []
    for row in range(n_rows):
        for col in range(n_cols):
            bl = row * cols + col
            br = row * cols + col + 1
            tl = (row + 1) * cols + col
            tr = (row + 1) * cols + col + 1
            fvc_list.append(4)
            fvi_list.extend([bl, br, tl, tr])
            u_min, u_max = col / n_cols, (col + 1) / n_cols
            v_min, v_max = row / n_rows, (row + 1) / n_rows
            uv_ranges.append([[u_min, v_min], [u_max, v_max]])

    return (points,
            np.array(fvc_list, dtype=np.int32),
            np.array(fvi_list, dtype=np.int32),
            np.array(uv_ranges, dtype=np.float64))


def make_boundary_lines(points, n_cols, n_rows):
    """Build line segments for the outer boundary of the fan mesh."""
    cols = n_cols + 1
    lines = []
    # Bottom arc (row 0)
    for c in range(n_cols):
        v0, v1 = c, c + 1
        lines.append([[points[v0][0], 0, points[v0][1]], [points[v1][0], 0, points[v1][1]]])
    # Top arc (last row)
    for c in range(n_cols):
        v0 = n_rows * cols + c
        v1 = n_rows * cols + c + 1
        lines.append([[points[v0][0], 0, points[v0][1]], [points[v1][0], 0, points[v1][1]]])
    # Left edge
    for r in range(n_rows):
        v0, v1 = r * cols, (r + 1) * cols
        lines.append([[points[v0][0], 0, points[v0][1]], [points[v1][0], 0, points[v1][1]]])
    # Right edge
    for r in range(n_rows):
        v0, v1 = r * cols + n_cols, (r + 1) * cols + n_cols
        lines.append([[points[v0][0], 0, points[v0][1]], [points[v1][0], 0, points[v1][1]]])
    return np.array(lines)


def add_subdiv_grid_lines(stage, face_quads, line_path='/geometry/SubdivGrid', color=(0.0, 1.0, 0.0)):
    """Draw quad edges for all sub-faces."""
    lines = []
    for q in face_quads:
        lines.append([[q[0][0], 0.0, q[0][1]], [q[1][0], 0.0, q[1][1]]])
        lines.append([[q[1][0], 0.0, q[1][1]], [q[3][0], 0.0, q[3][1]]])
        lines.append([[q[3][0], 0.0, q[3][1]], [q[2][0], 0.0, q[2][1]]])
        lines.append([[q[2][0], 0.0, q[2][1]], [q[0][0], 0.0, q[0][1]]])
    return io.add_lines_to_usd_stage(stage, np.array(lines), line_path, color=color)


# %%

# =========================================
# Cell 3: Single-quad bilinear baseline
# =========================================

from planar.BilinearInterpolationQuad import bilinear_interpolation_quad, reverse_bilinear_interpolation_quad

stage = io.load_usd_file(mesh_paths[0])
base_vertices = io.get_usd_mesh_vertices(stage, mesh_path)
base_vertices_xz = base_vertices[:, [0, 2]]

print("=== Single-Quad Bilinear Baseline ===")
bilinear_uv = bilinear_interpolation_quad(base_quad, base_vertices_xz)
baseline_vertices_xz = reverse_bilinear_interpolation_quad(target_quad, bilinear_uv)

new_vertices = np.column_stack((baseline_vertices_xz[:, 0], base_vertices[:, 1], baseline_vertices_xz[:, 1]))
stage = io.update_usd_mesh_vertices(stage, mesh_path, new_vertices)

baseline_output = str(output_dir / "Quad_TestVault_bilinear_baseline_output.usd")
stage.Export(baseline_output)
print(f"Saved: {baseline_output}")
open_usd_viewer(baseline_output)
del stage


# %%

# =========================================
# Cell 4: Fan mesh — bilinear subdivision level 2
# =========================================

from subdivision.BilinearSubdivision import bilinear_subdivide
from planar.SubdividedInterpolationQuad import subdivided_reverse_bilinear_interpolation_quad

N_COLS, N_ROWS = 2, 3
fan_points, fan_fvc, fan_fvi, fan_uv_ranges = make_fan_control_mesh(
    n_cols=N_COLS, n_rows=N_ROWS, r_inner=0.3, r_outer=0.9, angle_deg=70)

print("\n=== Fan Mesh — Bilinear Level 2 ===")
print(f"Control mesh: {len(fan_fvc)} faces, {len(fan_points)} vertices")

fan_subdiv = bilinear_subdivide(fan_points, fan_fvc, fan_fvi, levels=2,
                                 face_uv_ranges=fan_uv_ranges)
print(f"Sub-faces: {len(fan_subdiv['face_vertex_counts'])}")

stage = io.load_usd_file(mesh_paths[0])
base_vertices = io.get_usd_mesh_vertices(stage, mesh_path)
base_vertices_xz = base_vertices[:, [0, 2]]

bilinear_uv = bilinear_interpolation_quad(base_quad, base_vertices_xz)
target_vertices_xz = subdivided_reverse_bilinear_interpolation_quad(fan_subdiv, bilinear_uv)

deviation = np.linalg.norm(target_vertices_xz - baseline_vertices_xz, axis=1)
print(f"Max deviation from baseline: {np.max(deviation):.6f}")
print(f"Mean deviation from baseline: {np.mean(deviation):.6f}")

new_vertices = np.column_stack((target_vertices_xz[:, 0], base_vertices[:, 1], target_vertices_xz[:, 1]))
stage = io.update_usd_mesh_vertices(stage, mesh_path, new_vertices)

fan_boundary = make_boundary_lines(fan_points, N_COLS, N_ROWS)
stage = io.add_lines_to_usd_stage(stage, fan_boundary, '/geometry/FanBoundary', color=(1.0, 0.0, 0.0))
stage = add_subdiv_grid_lines(stage, fan_subdiv['face_quads'])

output_path = str(output_dir / "Quad_TestVault_fan_bilinear_subdiv2_output.usd")
stage.Export(output_path)
print(f"Saved: {output_path}")
open_usd_viewer(output_path)
del stage


# %%

# =========================================
# Cell 5: Fan mesh — Catmull-Clark subdivision level 2
# =========================================

from subdivision.CatmullClarkSubdivision import catmull_clark_subdivide

print("\n=== Fan Mesh — Catmull-Clark Level 2 ===")

fan_cc_subdiv = catmull_clark_subdivide(fan_points, fan_fvc, fan_fvi, levels=2,
                                         face_uv_ranges=fan_uv_ranges)
print(f"Sub-faces: {len(fan_cc_subdiv['face_vertex_counts'])}")

stage = io.load_usd_file(mesh_paths[0])
base_vertices = io.get_usd_mesh_vertices(stage, mesh_path)
base_vertices_xz = base_vertices[:, [0, 2]]

bilinear_uv = bilinear_interpolation_quad(base_quad, base_vertices_xz)
target_vertices_xz = subdivided_reverse_bilinear_interpolation_quad(fan_cc_subdiv, bilinear_uv)

deviation = np.linalg.norm(target_vertices_xz - baseline_vertices_xz, axis=1)
print(f"Max deviation from baseline: {np.max(deviation):.6f}")
print(f"Mean deviation from baseline: {np.mean(deviation):.6f}")

new_vertices = np.column_stack((target_vertices_xz[:, 0], base_vertices[:, 1], target_vertices_xz[:, 1]))
stage = io.update_usd_mesh_vertices(stage, mesh_path, new_vertices)

stage = io.add_lines_to_usd_stage(stage, fan_boundary, '/geometry/FanBoundary', color=(1.0, 0.0, 0.0))
stage = add_subdiv_grid_lines(stage, fan_cc_subdiv['face_quads'])

output_path = str(output_dir / "Quad_TestVault_fan_catmullclark_subdiv2_output.usd")
stage.Export(output_path)
print(f"Saved: {output_path}")
open_usd_viewer(output_path)

print("\n=== All tests complete ===")
del stage

# %%
