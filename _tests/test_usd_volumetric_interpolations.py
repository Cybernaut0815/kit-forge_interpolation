#%%

# =========================================
# Import necessary libraries
# =========================================

import sys
from pathlib import Path

# Add repository root to path so imports like "src.*" resolve correctly
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)

import numpy as np

import src.geometry.IO as io
from src.viz.usdview import open_usd_viewer

from importlib import reload


# %%

# =========================================
# Loading the geometry
# =========================================

input_dir = Path(__file__).resolve().parent / "input"

quad_high_poly_mesh_path = str(input_dir / "Quad_TestVault_highPoly.usd")
quad_low_poly_mesh_path = str(input_dir / "Quad_TestVault_lowPoly.usd")

tri_high_poly_mesh_path = str(input_dir / "Tri_TestVault_highPoly.usd")
tri_low_poly_mesh_path = str(input_dir / "Tri_TestVault_lowPoly.usd")

tetra_high_poly_mesh_path = str(input_dir / "Tetra_TestVault_highPoly.usd")
tetra_low_poly_mesh_path = str(input_dir / "Tetra_TestVault_lowPoly.usd")

penta_high_poly_mesh_path = str(input_dir / "Penta_TestVault_highPoly.usd")
penta_low_poly_mesh_path = str(input_dir / "Penta_TestVault_lowPoly.usd")

mesh_paths = [quad_high_poly_mesh_path, 
                quad_low_poly_mesh_path, 
                tri_high_poly_mesh_path, 
                tri_low_poly_mesh_path, 
                tetra_high_poly_mesh_path, 
                tetra_low_poly_mesh_path, 
                penta_high_poly_mesh_path, 
                penta_low_poly_mesh_path]

# %%

import src.interpolation.volumetric.TrilinearInterpolationHexahedron as triq
reload(triq)
from src.interpolation.volumetric.TrilinearInterpolationHexahedron import trilinear_interpolation_hexahedron, reverse_trilinear_interpolation_hexahedron

# =========================================
# Testing Trilinear Interpolation Hexahedron
# =========================================

mesh_index = 0

# Load USD stage
stage = io.load_usd_file(mesh_paths[mesh_index])

# Get mesh vertices
mesh_path = '/geometry/geometry'
base_vertices = io.get_usd_mesh_vertices(stage, mesh_path)

# Correct hexahedral vertex ordering: [v000, v100, v010, v110, v001, v101, v011, v111]
# For a box: x: -0.5 to 0.5, y: 0.0 to 1.0, z: -0.5 to 0.5
base_cube = np.array([
    [-0.5, 0.0, -0.5],  # v000: u=0, v=0, w=0
    [ 0.5, 0.0, -0.5],  # v100: u=1, v=0, w=0
    [-0.5, 1.0, -0.5],  # v010: u=0, v=1, w=0
    [ 0.5, 1.0, -0.5],  # v110: u=1, v=1, w=0
    [-0.5, 0.0,  0.5],  # v001: u=0, v=0, w=1
    [ 0.5, 0.0,  0.5],  # v101: u=1, v=0, w=1
    [-0.5, 1.0,  0.5],  # v011: u=0, v=1, w=1
    [ 0.5, 1.0,  0.5],  # v111: u=1, v=1, w=1
])

# Target cube with slight deformation (only moving one corner)
target_cube = np.array([
    [-0.5, 0.0, -0.5],  # v000: u=0, v=0, w=0
    [ 0.5, 0.0, -0.5],  # v100: u=1, v=0, w=0
    [-0.5, 1.0, -0.5],  # v010: u=0, v=1, w=0
    [ 0.5, 1.0, -0.5],  # v110: u=1, v=1, w=0
    [-0.5, 0.0,  0.5],  # v001: u=0, v=0, w=1
    [ 0.5, 0.0,  0.5],  # v101: u=1, v=0, w=1
    [-0.5, 1.0,  0.5],  # v011: u=0, v=1, w=1
    [ 0.25, 1.5,  0.25],  # v111: u=1, v=1, w=1 
])

# Debug: Test the cube mapping first
print("=== Debug Information ===")
print("Base cube vertices:")
for i, v in enumerate(base_cube):
    print(f"  v{i}: {v}")
print("\nTarget cube vertices:")
for i, v in enumerate(target_cube):
    print(f"  v{i}: {v}")

# Process the mesh
print(f"\nProcessing {len(base_vertices)} mesh vertices...")
trilinear_uvw = trilinear_interpolation_hexahedron(base_cube, base_vertices)
target_vertices_xyz = reverse_trilinear_interpolation_hexahedron(target_cube, trilinear_uvw)

print(f"Original vertices shape: {base_vertices.shape}")
print(f"UVW coordinates shape: {trilinear_uvw.shape}")
print(f"Target vertices shape: {target_vertices_xyz.shape}")

# Check for any extreme values that might indicate issues
print(f"UVW range: [{np.min(trilinear_uvw):.3f}, {np.max(trilinear_uvw):.3f}]")
print(f"Original XYZ range: [{np.min(base_vertices):.3f}, {np.max(base_vertices):.3f}]")
print(f"Target XYZ range: [{np.min(target_vertices_xyz):.3f}, {np.max(target_vertices_xyz):.3f}]")

new_vertices = target_vertices_xyz

# Update USD mesh vertices
stage = io.update_usd_mesh_vertices(stage, mesh_path, new_vertices)

# Prepare lines for visualization (x, 2, 3) format
# Convert hexahedral cube to line segments (12 edges)
cube_lines = np.array([
    # Bottom face (w=0)
    [target_cube[0], target_cube[1]],  # v000 to v100
    [target_cube[1], target_cube[3]],  # v100 to v110
    [target_cube[3], target_cube[2]],  # v110 to v010
    [target_cube[2], target_cube[0]],  # v010 to v000
    # Top face (w=1)
    [target_cube[4], target_cube[5]],  # v001 to v101
    [target_cube[5], target_cube[7]],  # v101 to v111
    [target_cube[7], target_cube[6]],  # v111 to v011
    [target_cube[6], target_cube[4]],  # v011 to v001
    # Vertical edges
    [target_cube[0], target_cube[4]],  # v000 to v001
    [target_cube[1], target_cube[5]],  # v100 to v101
    [target_cube[2], target_cube[6]],  # v010 to v011
    [target_cube[3], target_cube[7]],  # v110 to v111
])

# Add lines to USD stage (red color) under /geometry to inherit transform
stage = io.add_lines_to_usd_stage(stage, cube_lines, '/geometry/TargetCube', color=(1.0, 0.0, 0.0))

# Save the modified stage
output_path = str(output_dir / "Quad_TestVault_hexahedron_output.usd")
stage.Export(output_path)
print(f"\nSaved modified stage to: {output_path}")

# Open in USD viewer
print("\nOpening in USD viewer...")
open_usd_viewer(output_path)

# %%

# Delete stage to avoid chaining transforms
del stage

# =========================================
# Testing Barycentric Linear Interpolation Trigonal
# =========================================

import src.interpolation.volumetric.BarycentricLinearInterpolationTrigonal as blint
reload(blint)
from src.interpolation.volumetric.BarycentricLinearInterpolationTrigonal import barycentric_linear_interpolation_trigonal, reverse_barycentric_linear_interpolation_trigonal

mesh_index = 2

# Load USD stage
stage = io.load_usd_file(mesh_paths[mesh_index])

# Get mesh vertices
mesh_path = '/geometry/geometry'
base_vertices = io.get_usd_mesh_vertices(stage, mesh_path)

# Triangular prism vertex ordering: [v0_bottom, v1_bottom, v2_bottom, v0_top, v1_top, v2_top]
# Base triangular prism: equilateral triangle cross-section
# Bottom triangle at y=0.0, top triangle at y=1.0
base_prism = np.array([
    [0.0, 0.0, -0.57735],  # v0_bottom
    [-0.5, 0.0, 0.288663],  # v1_bottom
    [ 0.5, 0.0, 0.288663],  # v2_bottom
    [0.0, 1.0, -0.57735],  # v0_top
    [-0.5, 1.0, 0.288663],  # v1_top
    [ 0.5, 1.0, 0.288663],  # v2_top
])

# Target prism with deformation
target_prism = np.array([
    [0.0, 0.0, -0.7],      # v0_bottom
    [-0.6, 0.0, 0.35],     # v1_bottom
    [ 0.6, 0.0, 0.35],     # v2_bottom
    [0.1, 1.3, -0.5],      # v0_top
    [-0.5, 1.1, 0.4],      # v1_top
    [ 0.7, 1.2, 0.3],      # v2_top
])

# Debug
print("=== Debug Information ===")
print("Base prism vertices:")
for i, v in enumerate(base_prism):
    label = "bottom" if i < 3 else "top"
    print(f"  v{i} ({label}): {v}")
print("\nTarget prism vertices:")
for i, v in enumerate(target_prism):
    label = "bottom" if i < 3 else "top"
    print(f"  v{i} ({label}): {v}")

# Process the mesh
print(f"\nProcessing {len(base_vertices)} mesh vertices...")
parametric_coords = barycentric_linear_interpolation_trigonal(base_prism, base_vertices)
target_vertices_xyz = reverse_barycentric_linear_interpolation_trigonal(target_prism, parametric_coords)

print(f"Original vertices shape: {base_vertices.shape}")
print(f"Parametric coordinates shape: {parametric_coords.shape}")
print(f"Target vertices shape: {target_vertices_xyz.shape}")

# Check for any extreme values
print(f"Parametric range: [{np.min(parametric_coords):.3f}, {np.max(parametric_coords):.3f}]")
print(f"Barycentric range: [{np.min(parametric_coords[:, :3]):.3f}, {np.max(parametric_coords[:, :3]):.3f}]")
print(f"T range: [{np.min(parametric_coords[:, 3]):.3f}, {np.max(parametric_coords[:, 3]):.3f}]")
print(f"Original XYZ range: [{np.min(base_vertices):.3f}, {np.max(base_vertices):.3f}]")
print(f"Target XYZ range: [{np.min(target_vertices_xyz):.3f}, {np.max(target_vertices_xyz):.3f}]")

# Verify barycentric constraint
bary_sums = np.sum(parametric_coords[:, :3], axis=1)
print(f"Barycentric sum range: [{np.min(bary_sums):.6f}, {np.max(bary_sums):.6f}] (should be ~1.0)")

new_vertices = target_vertices_xyz

# Update USD mesh vertices
stage = io.update_usd_mesh_vertices(stage, mesh_path, new_vertices)

# Prepare lines for visualization
prism_lines = np.array([
    # Bottom triangle
    [target_prism[0], target_prism[1]],
    [target_prism[1], target_prism[2]],
    [target_prism[2], target_prism[0]],
    # Top triangle
    [target_prism[3], target_prism[4]],
    [target_prism[4], target_prism[5]],
    [target_prism[5], target_prism[3]],
    # Vertical edges
    [target_prism[0], target_prism[3]],
    [target_prism[1], target_prism[4]],
    [target_prism[2], target_prism[5]],
])

# Add lines under /geometry
stage = io.add_lines_to_usd_stage(stage, prism_lines, '/geometry/TargetPrism', color=(1.0, 0.0, 0.0))

# Save the modified stage
output_path = str(output_dir / "Tri_TestVault_prism_output.usd")
stage.Export(output_path)
print(f"\nSaved modified stage to: {output_path}")

# Open in USD viewer
print("\nOpening in USD viewer...")
open_usd_viewer(output_path)

# %%

# Delete stage to avoid chaining transforms
del stage

# =========================================
# Testing Barycentric Interpolation Tetrahedron
# =========================================

import src.interpolation.volumetric.BarycentricInterpolationTetrahedron as btet
reload(btet)
from src.interpolation.volumetric.BarycentricInterpolationTetrahedron import barycentric_interpolation_tetrahedron, reverse_barycentric_interpolation_tetrahedron

mesh_index = 4

# Load USD stage
stage = io.load_usd_file(mesh_paths[mesh_index])

# Get mesh vertices
mesh_path = '/geometry/geometry'
base_vertices = io.get_usd_mesh_vertices(stage, mesh_path)

# Tetrahedron vertex ordering: [v0, v1, v2, v3]
base_tetrahedron = np.array([
    [0.0, 0.0, -0.57735],  # v0_bottom
    [-0.5, 0.0, 0.288663],  # v1_bottom
    [ 0.5, 0.0, 0.288663],  # v2_bottom
    [ 0.0, 0.82, 0.0] # v3_top
])

# Target tetrahedron with deformation
target_tetrahedron = np.array([
    [0.0, 0.0, -0.57735],  # v0_bottom
    [-0.5, 0.0, 0.288663],  # v1_bottom
    [ 0.5, 0.0, 0.288663], # v2_bottom
    [ 0.2, 0.82, 0.2] # v3_top
])

# Debug
print("=== Debug Information ===")
print("Base tetrahedron vertices:")
for i, v in enumerate(base_tetrahedron):
    print(f"  v{i}: {v}")
print("\nTarget tetrahedron vertices:")
for i, v in enumerate(target_tetrahedron):
    print(f"  v{i}: {v}")

# Process the mesh
print(f"\nProcessing {len(base_vertices)} mesh vertices...")
barycentric_coords = barycentric_interpolation_tetrahedron(base_tetrahedron, base_vertices)
target_vertices_xyz = reverse_barycentric_interpolation_tetrahedron(target_tetrahedron, barycentric_coords)

print(f"Original vertices shape: {base_vertices.shape}")
print(f"Barycentric coordinates shape: {barycentric_coords.shape}")
print(f"Target vertices shape: {target_vertices_xyz.shape}")

# Check for any extreme values
print(f"Barycentric range: [{np.min(barycentric_coords):.3f}, {np.max(barycentric_coords):.3f}]")
print(f"Original XYZ range: [{np.min(base_vertices):.3f}, {np.max(base_vertices):.3f}]")
print(f"Target XYZ range: [{np.min(target_vertices_xyz):.3f}, {np.max(target_vertices_xyz):.3f}]")

# Verify barycentric constraint
bary_sums = np.sum(barycentric_coords, axis=1)
print(f"Barycentric sum range: [{np.min(bary_sums):.6f}, {np.max(bary_sums):.6f}] (should be ~1.0)")

new_vertices = target_vertices_xyz

# Update USD mesh vertices
stage = io.update_usd_mesh_vertices(stage, mesh_path, new_vertices)

# Prepare lines for visualization
tetrahedron_lines = np.array([
    # Base triangle edges
    [target_tetrahedron[1], target_tetrahedron[2]],  # v1 to v2
    [target_tetrahedron[2], target_tetrahedron[3]],  # v2 to v3
    [target_tetrahedron[3], target_tetrahedron[1]],  # v3 to v1
    # Edges from apex to base
    [target_tetrahedron[0], target_tetrahedron[1]],  # v0 to v1
    [target_tetrahedron[0], target_tetrahedron[2]],  # v0 to v2
    [target_tetrahedron[0], target_tetrahedron[3]],  # v0 to v3
])

# Add lines under /geometry
stage = io.add_lines_to_usd_stage(stage, tetrahedron_lines, '/geometry/TargetTetrahedron', color=(1.0, 0.0, 0.0))

# Save the modified stage
output_path = str(output_dir / "Tetra_TestVault_tetrahedron_output.usd")
stage.Export(output_path)
print(f"\nSaved modified stage to: {output_path}")

# Open in USD viewer
print("\nOpening in USD viewer...")
open_usd_viewer(output_path)

# %%

# Delete stage to avoid chaining transforms
del stage

# =========================================
# Testing Trilinear Interpolation Pentahedron
# =========================================

import src.interpolation.volumetric.TrilinearInterpolationPentahedron as penta
reload(penta)
from src.interpolation.volumetric.TrilinearInterpolationPentahedron import trilinear_interpolation_pentahedron, reverse_trilinear_interpolation_pentahedron

mesh_index = 6

# Load USD stage
stage = io.load_usd_file(mesh_paths[mesh_index])

# Get mesh vertices
mesh_path = '/geometry/geometry'
base_vertices = io.get_usd_mesh_vertices(stage, mesh_path)

# Pentahedron (quadrangular pyramid) vertex ordering: [v00, v10, v01, v11, apex]
base_pyramid = np.array([
    [-0.5, 0.0, -0.5],  # v00: base bottom-left
    [ 0.5, 0.0, -0.5],  # v10: base bottom-right
    [-0.5, 0.0,  0.5],  # v01: base top-left
    [ 0.5, 0.0,  0.5],  # v11: base top-right
    [ 0.0, 0.82,  0.0],  # apex: pyramid tip 
])

# Target pyramid with deformation
target_pyramid = np.array([
    [-0.5, 0.0, -0.5],  # v00: base bottom-left
    [ 0.5, 0.0, -0.5],  # v10: base bottom-right
    [-0.5, 0.0,  0.5],  # v01: base top-left
    [ 0.6, 0.0,  0.7],  # v11: base top-right
    [ 0.2, 1.26,  0.2],  # apex: pyramid tip (deformed)
])

# Debug
print("=== Debug Information ===")
print("Base pyramid vertices:")
for i, v in enumerate(base_pyramid):
    label = "apex" if i == 4 else f"base v{i}"
    print(f"  {label}: {v}")
print("\nTarget pyramid vertices:")
for i, v in enumerate(target_pyramid):
    label = "apex" if i == 4 else f"base v{i}"
    print(f"  {label}: {v}")

# Process the mesh
print(f"\nProcessing {len(base_vertices)} mesh vertices...")
uvw_coords = trilinear_interpolation_pentahedron(base_pyramid, base_vertices)
target_vertices_xyz = reverse_trilinear_interpolation_pentahedron(target_pyramid, uvw_coords)

print(f"Original vertices shape: {base_vertices.shape}")
print(f"UVW coordinates shape: {uvw_coords.shape}")
print(f"Target vertices shape: {target_vertices_xyz.shape}")

# Check for any extreme values
print(f"UVW range: [{np.min(uvw_coords):.3f}, {np.max(uvw_coords):.3f}]")
print(f"Original XYZ range: [{np.min(base_vertices):.3f}, {np.max(base_vertices):.3f}]")
print(f"Target XYZ range: [{np.min(target_vertices_xyz):.3f}, {np.max(target_vertices_xyz):.3f}]")

new_vertices = target_vertices_xyz

# Update USD mesh vertices
stage = io.update_usd_mesh_vertices(stage, mesh_path, new_vertices)

# Prepare lines for visualization
pyramid_lines = np.array([
    # Base quadrilateral edges
    [target_pyramid[0], target_pyramid[1]],  # v00 to v10
    [target_pyramid[1], target_pyramid[3]],  # v10 to v11
    [target_pyramid[3], target_pyramid[2]],  # v11 to v01
    [target_pyramid[2], target_pyramid[0]],  # v01 to v00
    # Edges from base corners to apex
    [target_pyramid[0], target_pyramid[4]],  # v00 to apex
    [target_pyramid[1], target_pyramid[4]],  # v10 to apex
    [target_pyramid[2], target_pyramid[4]],  # v01 to apex
    [target_pyramid[3], target_pyramid[4]],  # v11 to apex
])

# Add lines under /geometry
stage = io.add_lines_to_usd_stage(stage, pyramid_lines, '/geometry/TargetPyramid', color=(1.0, 0.0, 0.0))

# Save the modified stage
output_path = str(output_dir / "Penta_TestVault_pyramid_output.usd")
stage.Export(output_path)
print(f"\nSaved modified stage to: {output_path}")

# Open in USD viewer
print("\nOpening in USD viewer...")
open_usd_viewer(output_path)

# %%
