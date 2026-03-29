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

mesh_paths = [quad_high_poly_mesh_path, quad_low_poly_mesh_path, tri_high_poly_mesh_path, tri_low_poly_mesh_path]


# %%

# =========================================
# Testing bilinear interpolation
# =========================================

from src.interpolation.planar.BilinearInterpolationQuad import bilinear_interpolation_quad, reverse_bilinear_interpolation_quad

# Test for high poly quad mesh
mesh_index = 0

# Load USD stage
stage = io.load_usd_file(mesh_paths[mesh_index])

# Get mesh vertices - assuming mesh is at /World/Mesh
mesh_path = '/geometry/geometry'
base_vertices = io.get_usd_mesh_vertices(stage, mesh_path)

base_quad = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
target_quad = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.7, 0.7], [0.5, 0.5]])

# The mesh geometry is Y-up (ground plane is XZ), despite stage metadata
# Work in XZ plane (ground plane), keeping Y (height) unchanged
base_vertices_xz = base_vertices[:,[0,2]]

# Debug: Test the quad mapping first
print("=== Bilinear Interpolation Debug ===")
print("Base quad vertices:")
for i, v in enumerate(base_quad):
    print(f"  v{i}: {v}")
print("\nTarget quad vertices:")
for i, v in enumerate(target_quad):
    print(f"  v{i}: {v}")

# Process the mesh
print(f"\nProcessing {len(base_vertices_xz)} mesh vertices...")
bilinear_uv = bilinear_interpolation_quad(base_quad, base_vertices_xz)
target_vertices_xz = reverse_bilinear_interpolation_quad(target_quad, bilinear_uv)

print(f"Original XZ vertices shape: {base_vertices_xz.shape}")
print(f"UV coordinates shape: {bilinear_uv.shape}")
print(f"Target XZ vertices shape: {target_vertices_xz.shape}")

# Check for any extreme values that might indicate issues
print(f"UV range: [{np.min(bilinear_uv):.3f}, {np.max(bilinear_uv):.3f}]")
print(f"Original XZ range: [{np.min(base_vertices_xz):.3f}, {np.max(base_vertices_xz):.3f}]")
print(f"Target XZ range: [{np.min(target_vertices_xz):.3f}, {np.max(target_vertices_xz):.3f}]")

# Reconstruct 3D vertices: keep Y (height) unchanged
new_vertices = np.column_stack((target_vertices_xz[:,0], base_vertices[:,1], target_vertices_xz[:,1]))

# Update USD mesh vertices
stage = io.update_usd_mesh_vertices(stage, mesh_path, new_vertices)

# Prepare lines for visualization (N, 2, 3) format
# Mesh is Y-up (ground plane is XZ), so lines in XZ plane at y=0
# target_quad contains [X, Z] values, so we map to [X, Y=0, Z]
quad_lines = np.array([
    [[target_quad[0][0], 0.0, target_quad[0][1]], [target_quad[1][0], 0.0, target_quad[1][1]]],  # v0 to v1
    [[target_quad[1][0], 0.0, target_quad[1][1]], [target_quad[3][0], 0.0, target_quad[3][1]]],  # v1 to v3
    [[target_quad[3][0], 0.0, target_quad[3][1]], [target_quad[2][0], 0.0, target_quad[2][1]]],  # v3 to v2
    [[target_quad[2][0], 0.0, target_quad[2][1]], [target_quad[0][0], 0.0, target_quad[0][1]]],  # v2 to v0
])

# Add lines to USD stage (red color)
# Add under /geometry so they inherit the Y-up to Z-up transform
stage = io.add_lines_to_usd_stage(stage, quad_lines, '/geometry/TargetQuad', color=(1.0, 0.0, 0.0))

# Save the modified stage
output_path = str(output_dir / "Quad_TestVault_bilinear_output.usd")
stage.Export(output_path)
print(f"\nSaved modified stage to: {output_path}")

# Open in USD viewer
print("\nOpening in USD viewer...")
open_usd_viewer(output_path)


# %%

# stage from before needs to be deleted to not chain transforms
del stage

# =========================================
# Testing projective interpolation
# =========================================

from src.interpolation.planar.ProjectiveInterpolationQuad import projective_interpolation_quad, reverse_projective_interpolation_quad

# Test for high poly quad mesh
mesh_index = 0

# Load USD stage
stage = io.load_usd_file(mesh_paths[mesh_index])

# Get mesh vertices
mesh_path = '/geometry/geometry'
base_vertices = io.get_usd_mesh_vertices(stage, mesh_path)

base_quad = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
target_quad = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.7, 0.7], [0.5, 0.5]])

# The mesh geometry is Y-up (ground plane is XZ)
# Work in XZ plane (ground plane), keeping Y (height) unchanged
base_vertices_xz = base_vertices[:,[0,2]]

# Debug: Test the quad mapping first
print("=== Projective Interpolation Debug ===")
print("Base quad vertices:")
for i, v in enumerate(base_quad):
    print(f"  v{i}: {v}")
print("\nTarget quad vertices:")
for i, v in enumerate(target_quad):
    print(f"  v{i}: {v}")

# Process the mesh
print(f"\nProcessing {len(base_vertices_xz)} mesh vertices...")
projective_uv = projective_interpolation_quad(base_quad, base_vertices_xz)
target_vertices_xz = reverse_projective_interpolation_quad(target_quad, projective_uv)

print(f"Original XZ vertices shape: {base_vertices_xz.shape}")
print(f"UV coordinates shape: {projective_uv.shape}")
print(f"Target XZ vertices shape: {target_vertices_xz.shape}")

# Check for any extreme values that might indicate issues
print(f"UV range: [{np.min(projective_uv):.3f}, {np.max(projective_uv):.3f}]")
print(f"Original XZ range: [{np.min(base_vertices_xz):.3f}, {np.max(base_vertices_xz):.3f}]")
print(f"Target XZ range: [{np.min(target_vertices_xz):.3f}, {np.max(target_vertices_xz):.3f}]")

# Reconstruct 3D vertices: keep Y (height) unchanged
new_vertices = np.column_stack((target_vertices_xz[:,0], base_vertices[:,1], target_vertices_xz[:,1]))

# Update USD mesh vertices
stage = io.update_usd_mesh_vertices(stage, mesh_path, new_vertices)

# Prepare lines for visualization
# Mesh is Y-up (ground plane is XZ), so lines in XZ plane at y=0
# target_quad contains [X, Z] values, so we map to [X, Y=0, Z]
quad_lines = np.array([
    [[target_quad[0][0], 0.0, target_quad[0][1]], [target_quad[1][0], 0.0, target_quad[1][1]]],  # v0 to v1
    [[target_quad[1][0], 0.0, target_quad[1][1]], [target_quad[3][0], 0.0, target_quad[3][1]]],  # v1 to v3
    [[target_quad[3][0], 0.0, target_quad[3][1]], [target_quad[2][0], 0.0, target_quad[2][1]]],  # v3 to v2
    [[target_quad[2][0], 0.0, target_quad[2][1]], [target_quad[0][0], 0.0, target_quad[0][1]]],  # v2 to v0
])

# Add lines to USD stage (red color)
# Add under /geometry so they inherit the Y-up to Z-up transform
stage = io.add_lines_to_usd_stage(stage, quad_lines, '/geometry/TargetQuad', color=(1.0, 0.0, 0.0))

# Save the modified stage
output_path = str(output_dir / "Quad_TestVault_projective_output.usd")
stage.Export(output_path)
print(f"\nSaved modified stage to: {output_path}")

# Open in USD viewer
print("\nOpening in USD viewer...")
open_usd_viewer(output_path)


# %%

# stage from before needs to be deleted to not chain transforms
del stage

# =========================================
# Testing barycentric interpolation
# =========================================

from src.interpolation.planar.BarycentricInterpolationTri import barycentric_interpolation_tri, reverse_barycentric_interpolation_tri

# Test for high poly tri mesh
mesh_index = 2

# Load USD stage
stage = io.load_usd_file(mesh_paths[mesh_index])

# Get mesh vertices
mesh_path = '/geometry/geometry'
base_vertices = io.get_usd_mesh_vertices(stage, mesh_path)

base_tri = np.array([[0, -0.57735], [-0.5, 0.288663], [0.5, 0.288663]])
target_tri = np.array([[0, -0.57735], [-0.5, 0.57735], [0.5, 0.288663]])

# The mesh geometry is Y-up (ground plane is XZ)
# Work in XZ plane (ground plane), keeping Y (height) unchanged
base_vertices_xz = base_vertices[:,[0,2]]

# Debug: Test the triangle mapping first
print("=== Barycentric Interpolation Debug ===")
print("Base triangle vertices:")
for i, v in enumerate(base_tri):
    print(f"  v{i}: {v}")
print("\nTarget triangle vertices:")
for i, v in enumerate(target_tri):
    print(f"  v{i}: {v}")

# Process the mesh
print(f"\nProcessing {len(base_vertices_xz)} mesh vertices...")
barycentric_uv = barycentric_interpolation_tri(base_tri, base_vertices_xz)
target_vertices_xz = reverse_barycentric_interpolation_tri(target_tri, barycentric_uv)

print(f"Original XZ vertices shape: {base_vertices_xz.shape}")
print(f"Barycentric coordinates shape: {barycentric_uv.shape}")
print(f"Target XZ vertices shape: {target_vertices_xz.shape}")

# Check for any extreme values that might indicate issues
print(f"Barycentric range: [{np.min(barycentric_uv):.3f}, {np.max(barycentric_uv):.3f}]")
print(f"Original XZ range: [{np.min(base_vertices_xz):.3f}, {np.max(base_vertices_xz):.3f}]")
print(f"Target XZ range: [{np.min(target_vertices_xz):.3f}, {np.max(target_vertices_xz):.3f}]")

# Reconstruct 3D vertices: keep Y (height) unchanged
new_vertices = np.column_stack((target_vertices_xz[:,0], base_vertices[:,1], target_vertices_xz[:,1]))

# Update USD mesh vertices
stage = io.update_usd_mesh_vertices(stage, mesh_path, new_vertices)

# Prepare lines for visualization
# Mesh is Y-up (ground plane is XZ), so lines in XZ plane at y=0
# target_tri contains [X, Z] values, so we map to [X, Y=0, Z]
tri_lines = np.array([
    [[target_tri[0][0], 0.0, target_tri[0][1]], [target_tri[1][0], 0.0, target_tri[1][1]]],  # v0 to v1
    [[target_tri[1][0], 0.0, target_tri[1][1]], [target_tri[2][0], 0.0, target_tri[2][1]]],  # v1 to v2
    [[target_tri[2][0], 0.0, target_tri[2][1]], [target_tri[0][0], 0.0, target_tri[0][1]]],  # v2 to v0
])

# Add lines to USD stage (red color)
# Add under /geometry so they inherit the Y-up to Z-up transform
stage = io.add_lines_to_usd_stage(stage, tri_lines, '/geometry/TargetTriangle', color=(1.0, 0.0, 0.0))

# Save the modified stage
output_path = str(output_dir / "Tri_TestVault_barycentric_output.usd")
stage.Export(output_path)
print(f"\nSaved modified stage to: {output_path}")

# Open in USD viewer
print("\nOpening in USD viewer...")
open_usd_viewer(output_path)

# %%
