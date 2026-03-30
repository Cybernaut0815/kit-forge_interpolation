import numpy as np

try:
    from planar.BilinearInterpolationQuad import (
        bilinear_interpolation_quad,
        reverse_bilinear_interpolation_quad,
        reverse_bilinear_interpolation_quad_with_tangents,
    )
except ModuleNotFoundError:
    from src.interpolation.planar.BilinearInterpolationQuad import (
        bilinear_interpolation_quad,
        reverse_bilinear_interpolation_quad,
        reverse_bilinear_interpolation_quad_with_tangents,
    )


def _find_sub_face(uv_point, face_uv_ranges):
    """
    Find the sub-face containing a UV point via brute-force search.

    Args:
        uv_point (np.array): Shape (2,) — UV coordinate to locate.
        face_uv_ranges (np.array): Shape (F, 2, 2) — [[u_min, v_min], [u_max, v_max]] per face.

    Returns:
        int: Index of the sub-face containing the point, or -1 if not found.
    """
    u, v = uv_point
    eps = 1e-10
    for i in range(len(face_uv_ranges)):
        u_min, v_min = face_uv_ranges[i, 0]
        u_max, v_max = face_uv_ranges[i, 1]
        if u_min - eps <= u <= u_max + eps and v_min - eps <= v <= v_max + eps:
            return i
    return -1


def _global_to_local_uv(uv_point, uv_range):
    """
    Convert a global UV [0,1]² coordinate to local UV within a sub-face.

    Args:
        uv_point (np.array): Shape (2,) — global UV coordinate.
        uv_range (np.array): Shape (2, 2) — [[u_min, v_min], [u_max, v_max]].

    Returns:
        np.array: Shape (2,) — local UV in [0,1]².
    """
    u_min, v_min = uv_range[0]
    u_max, v_max = uv_range[1]
    u_size = u_max - u_min
    v_size = v_max - v_min
    local_u = (uv_point[0] - u_min) / u_size if u_size > 0 else 0.5
    local_v = (uv_point[1] - v_min) / v_size if v_size > 0 else 0.5
    return np.array([
        np.clip(local_u, 0.0, 1.0),
        np.clip(local_v, 0.0, 1.0),
    ])


def _local_to_global_uv(local_uv, uv_range):
    """
    Convert a local UV [0,1]² coordinate back to global UV within the parent range.

    Args:
        local_uv (np.array): Shape (2,) — local UV in [0,1]².
        uv_range (np.array): Shape (2, 2) — [[u_min, v_min], [u_max, v_max]].

    Returns:
        np.array: Shape (2,) — global UV coordinate.
    """
    u_min, v_min = uv_range[0]
    u_max, v_max = uv_range[1]
    return np.array([
        u_min + local_uv[0] * (u_max - u_min),
        v_min + local_uv[1] * (v_max - v_min),
    ])


def subdivided_bilinear_interpolation_quad(subdiv_result, cartesian_coordinates):
    """
    Forward interpolation: Cartesian → global UV using a subdivided quad mesh.

    For each input point, finds the containing sub-face and computes the global
    UV coordinate by first computing the local UV within the sub-face's quad and
    then mapping it to the sub-face's global UV range.

    Args:
        subdiv_result (dict): Output of bilinear_subdivide() or catmull_clark_subdivide().
            Must contain 'face_quads' and 'face_uv_ranges'.
        cartesian_coordinates (np.array): Shape (N, 2) — Cartesian points to map.

    Returns:
        np.array: Shape (N, 2) — Global UV coordinates in [0,1]², or None on failure.
    """
    face_quads = subdiv_result['face_quads']
    face_uv_ranges = subdiv_result['face_uv_ranges']

    cartesian_coordinates = np.atleast_2d(cartesian_coordinates)
    num_points = len(cartesian_coordinates)
    num_faces = len(face_quads)
    result_uv = np.zeros((num_points, 2), dtype=np.float64)

    for p in range(num_points):
        pt = cartesian_coordinates[p]
        best_face = -1
        best_local_uv = None

        # Try each sub-face and pick the one where the local UV is closest to [0,1]
        for f in range(num_faces):
            quad = face_quads[f]
            local_uv = bilinear_interpolation_quad(quad, pt[np.newaxis, :])
            if local_uv is None:
                continue
            local_uv = local_uv[0]
            # Check if inside [0,1]² (with tolerance)
            if (-0.01 <= local_uv[0] <= 1.01) and (-0.01 <= local_uv[1] <= 1.01):
                best_face = f
                best_local_uv = np.clip(local_uv, 0.0, 1.0)
                if (0.0 <= local_uv[0] <= 1.0) and (0.0 <= local_uv[1] <= 1.0):
                    break  # Exact match, stop searching

        if best_face >= 0:
            result_uv[p] = _local_to_global_uv(best_local_uv, face_uv_ranges[best_face])
        else:
            result_uv[p] = np.array([np.nan, np.nan])

    return result_uv


def subdivided_reverse_bilinear_interpolation_quad(subdiv_result, uv):
    """
    Reverse interpolation: global UV → Cartesian using a subdivided quad mesh.

    For each input UV, locates the containing sub-face, converts to local UV,
    then bilinearly interpolates within that sub-face's quad geometry.

    Args:
        subdiv_result (dict): Output of bilinear_subdivide() or catmull_clark_subdivide().
            Must contain 'face_quads' and 'face_uv_ranges'.
        uv (np.array): Shape (N, 2) — Global UV coordinates in [0,1]².

    Returns:
        np.array: Shape (N, 2) — Cartesian coordinates, or None on failure.
    """
    face_quads = subdiv_result['face_quads']
    face_uv_ranges = subdiv_result['face_uv_ranges']

    uv = np.atleast_2d(uv)
    num_points = len(uv)
    dim = face_quads.shape[2]
    result = np.zeros((num_points, dim), dtype=np.float64)

    for p in range(num_points):
        face_idx = _find_sub_face(uv[p], face_uv_ranges)
        if face_idx < 0:
            result[p] = np.nan
            continue
        local_uv = _global_to_local_uv(uv[p], face_uv_ranges[face_idx])
        quad = face_quads[face_idx]
        cart = reverse_bilinear_interpolation_quad(quad, local_uv[np.newaxis, :])
        if cart is None:
            result[p] = np.nan
        else:
            result[p] = cart[0]

    return result


def subdivided_reverse_bilinear_interpolation_quad_with_tangents(subdiv_result, uv):
    """
    Reverse interpolation with tangents: global UV → Cartesian + tangent vectors.

    For each input UV, locates the containing sub-face, converts to local UV,
    then computes both the position and tangent vectors within that sub-face's
    quad geometry.

    Args:
        subdiv_result (dict): Output of bilinear_subdivide() or catmull_clark_subdivide().
            Must contain 'face_quads' and 'face_uv_ranges'.
        uv (np.array): Shape (N, 2) — Global UV coordinates in [0,1]².

    Returns:
        tuple: (cartesian_coords, tangent_u, tangent_v)
            - cartesian_coords: Shape (N, 2) — Cartesian positions
            - tangent_u: Shape (N, 2) — Tangent vectors in U direction
            - tangent_v: Shape (N, 2) — Tangent vectors in V direction
            Returns (None, None, None) on failure.
    """
    face_quads = subdiv_result['face_quads']
    face_uv_ranges = subdiv_result['face_uv_ranges']

    uv = np.atleast_2d(uv)
    num_points = len(uv)
    dim = face_quads.shape[2]
    result_pos = np.zeros((num_points, dim), dtype=np.float64)
    result_tu = np.zeros((num_points, dim), dtype=np.float64)
    result_tv = np.zeros((num_points, dim), dtype=np.float64)

    for p in range(num_points):
        face_idx = _find_sub_face(uv[p], face_uv_ranges)
        if face_idx < 0:
            result_pos[p] = np.nan
            result_tu[p] = np.nan
            result_tv[p] = np.nan
            continue
        local_uv = _global_to_local_uv(uv[p], face_uv_ranges[face_idx])
        quad = face_quads[face_idx]
        cart, tu, tv = reverse_bilinear_interpolation_quad_with_tangents(
            quad, local_uv[np.newaxis, :])
        if cart is None:
            result_pos[p] = np.nan
            result_tu[p] = np.nan
            result_tv[p] = np.nan
        else:
            result_pos[p] = cart[0]
            result_tu[p] = tu[0]
            result_tv[p] = tv[0]

    return result_pos, result_tu, result_tv
