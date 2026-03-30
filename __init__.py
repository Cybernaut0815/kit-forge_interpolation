# Interpolation package - geometric interpolation methods for 1D, 2D, and 3D elements

# Linear (1D) interpolation
from .linear import (
    linear_interpolation_line,
    reverse_linear_interpolation_line,
    reverse_linear_interpolation_line_with_tangent
)

# Planar (2D) interpolation
from .planar import (
    # Triangle
    barycentric_interpolation_tri,
    reverse_barycentric_interpolation_tri,
    reverse_barycentric_interpolation_tri_with_tangents,
    # Quadrilateral - Bilinear
    bilinear_interpolation_quad,
    reverse_bilinear_interpolation_quad,
    reverse_bilinear_interpolation_quad_with_tangents,
    # Quadrilateral - Projective
    projective_interpolation_quad,
    reverse_projective_interpolation_quad,
    reverse_projective_interpolation_quad_with_tangents,
    # Quadrilateral - Subdivided
    subdivided_bilinear_interpolation_quad,
    subdivided_reverse_bilinear_interpolation_quad,
    subdivided_reverse_bilinear_interpolation_quad_with_tangents
)

# Subdivision
from .subdivision import (
    bilinear_subdivide,
    catmull_clark_subdivide
)

# Volumetric (3D) interpolation
from .volumetric import (
    # Tetrahedron
    barycentric_interpolation_tetrahedron,
    reverse_barycentric_interpolation_tetrahedron,
    reverse_barycentric_interpolation_tetrahedron_with_tangents,
    # Triangular Prism
    barycentric_linear_interpolation_trigonal,
    reverse_barycentric_linear_interpolation_trigonal,
    reverse_barycentric_linear_interpolation_trigonal_with_tangents,
    # Hexahedron
    trilinear_interpolation_hexahedron,
    reverse_trilinear_interpolation_hexahedron,
    reverse_trilinear_interpolation_hexahedron_with_tangents,
    # Pentahedron (Square Pyramid)
    trilinear_interpolation_pentahedron,
    reverse_trilinear_interpolation_pentahedron,
    reverse_trilinear_interpolation_pentahedron_with_tangents
)

__all__ = [
    # Linear (1D)
    'linear_interpolation_line',
    'reverse_linear_interpolation_line',
    'reverse_linear_interpolation_line_with_tangent',
    # Planar (2D) - Triangle
    'barycentric_interpolation_tri',
    'reverse_barycentric_interpolation_tri',
    'reverse_barycentric_interpolation_tri_with_tangents',
    # Planar (2D) - Quadrilateral
    'bilinear_interpolation_quad',
    'reverse_bilinear_interpolation_quad',
    'reverse_bilinear_interpolation_quad_with_tangents',
    'projective_interpolation_quad',
    'reverse_projective_interpolation_quad',
    'reverse_projective_interpolation_quad_with_tangents',
    # Planar (2D) - Subdivided Quadrilateral
    'subdivided_bilinear_interpolation_quad',
    'subdivided_reverse_bilinear_interpolation_quad',
    'subdivided_reverse_bilinear_interpolation_quad_with_tangents',
    # Subdivision
    'bilinear_subdivide',
    'catmull_clark_subdivide',
    # Volumetric (3D) - Tetrahedron
    'barycentric_interpolation_tetrahedron',
    'reverse_barycentric_interpolation_tetrahedron',
    'reverse_barycentric_interpolation_tetrahedron_with_tangents',
    # Volumetric (3D) - Triangular Prism
    'barycentric_linear_interpolation_trigonal',
    'reverse_barycentric_linear_interpolation_trigonal',
    'reverse_barycentric_linear_interpolation_trigonal_with_tangents',
    # Volumetric (3D) - Hexahedron
    'trilinear_interpolation_hexahedron',
    'reverse_trilinear_interpolation_hexahedron',
    'reverse_trilinear_interpolation_hexahedron_with_tangents',
    # Volumetric (3D) - Pentahedron
    'trilinear_interpolation_pentahedron',
    'reverse_trilinear_interpolation_pentahedron',
    'reverse_trilinear_interpolation_pentahedron_with_tangents',
]

