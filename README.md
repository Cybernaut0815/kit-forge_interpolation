# Interpolation Submodule

Mathematical interpolation kernels for 1D, 2D, and 3D geometry deformation workflows.

This submodule contains vectorized interpolation methods that map between Cartesian coordinates and parametric spaces, then reconstruct transformed coordinates in target domains. It is designed as a reusable computation core for geometry tools, mesh pipelines, and WFC-driven geometry systems.

## Scope

The module currently includes:

- 1D line interpolation
- 2D planar interpolation for quads and triangles
- 3D volumetric interpolation for hexahedra, triangular prisms, tetrahedra, and pentahedra
- Tangent/direction output helpers for differential behavior and visualization
- Development test scripts for mathematical behavior and USD-based visual debugging

## Folder Layout

- `helper.py`: shared math utilities (`wedge_2d`, `lerp`, validation helpers, UV grid generation)
- `linear/LinearInterpolationLine.py`: line forward/reverse interpolation and tangent extraction
- `planar/BilinearInterpolationQuad.py`: bilinear quad forward/reverse interpolation (+ tangents)
- `planar/ProjectiveInterpolationQuad.py`: projective quad forward/reverse interpolation (+ tangents)
- `planar/BarycentricInterpolationTri.py`: triangle barycentric forward/reverse interpolation (+ tangents)
- `volumetric/TrilinearInterpolationHexahedron.py`: hexahedral trilinear forward/reverse interpolation (+ tangents)
- `volumetric/BarycentricLinearInterpolationTrigonal.py`: triangular prism interpolation with barycentric-in-plane plus linear depth (+ tangents)
- `volumetric/BarycentricInterpolationTetrahedron.py`: tetrahedral barycentric forward/reverse interpolation (+ direction vectors)
- `volumetric/TrilinearInterpolationPentahedron.py`: pentahedral interpolation (+ tangents)
- `_tests/`: executable scripts for validation and USD output generation
- `_tests/output/`: generated USD files from test runs

## Concepts and Conventions

### Forward vs reverse mapping

Most implementations follow a pair of operations:

- Forward interpolation: Cartesian -> parametric coordinates (`t`, `uv`, `uvw`, barycentric)
- Reverse interpolation: parametric -> Cartesian coordinates in a target shape

This enables transferring embedded points from a base primitive to a deformed target primitive while preserving parametric position.

### Tangents and directional derivatives

Several reverse methods include tangent/direction outputs. These vectors can be used for:

- local orientation frames
- anisotropic placement
- deformation diagnostics
- visual debugging in USD

### Vectorization

Implementations are NumPy-vectorized to process many points at once and are intended for mesh-scale coordinate transfer.

## Dependency Profile

### Core math usage

For pure interpolation math kernels, the essential dependencies are:

- `numpy`
- `scipy`

### Integration and visualization usage

Some development tests and visualization scripts also rely on:

- `usd-core` (OpenUSD Python bindings)
- `trimesh`
- `pillow`
- optional geometry ecosystem packages used elsewhere in the host project

## Standalone Setup (Submodule-only)

If you want to use and develop this submodule independently of the parent repository:

1. Clone the submodule repository:

```bash
git clone https://github.com/Cybernaut0815/kit-forge_interpolation.git
cd kit-forge_interpolation
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows cmd:

```bat
.venv\Scripts\activate.bat
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install numpy scipy
```

Optional integration stack for USD-oriented scripts:

```bash
pip install usd-core trimesh pillow
```

## Standalone Development Workflow

### Import examples

```python
import numpy as np
from planar.BilinearInterpolationQuad import bilinear_interpolation_quad, reverse_bilinear_interpolation_quad

base_quad = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
target_quad = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.7, 0.7], [0.5, 0.5]])
points = np.array([[0.0, 0.0], [0.2, 0.1]])

uv = bilinear_interpolation_quad(base_quad, points)
mapped = reverse_bilinear_interpolation_quad(target_quad, uv)
```

### Running tests/scripts

Current `_tests` scripts are development scripts rather than strict unit tests. Two groups exist:

- math-only scripts (no USD stage output)
- USD integration scripts that load meshes and open viewers

Run from repository root:

```bash
python _tests/test_linear_interpolations.py
```

For USD-producing scripts, ensure:

- required USD dependencies are installed
- expected asset paths exist in your working copy

## Using as a Submodule in another Repo

To consume this module in another project:

```bash
git submodule add https://github.com/Cybernaut0815/kit-forge_interpolation.git src/interpolation
git submodule update --init --recursive
```

Typical update flow:

```bash
cd src/interpolation
git checkout main
git pull
cd ../..
git add src/interpolation
git commit -m "Update interpolation submodule"
```

## Quality Notes and Roadmap

Planned improvements for this submodule:

- migrate development scripts into formal `pytest` suites
- separate pure-math tests from host-project integration tests
- add typed public API wrappers and versioned releases
- provide benchmark cases for large mesh batches

## License and Ownership

This repository is part of the GeometryDevEnv ecosystem and is intended to be reusable as an independent interpolation toolkit.