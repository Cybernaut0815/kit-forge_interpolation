import numpy as np


def bilinear_subdivide(points, face_vertex_counts, face_vertex_indices, levels=1,
                       face_uv_ranges=None):
    """
    Perform bilinear (uniform midpoint) subdivision on a quad mesh.

    Each quad face is split into 4 sub-quads by inserting edge midpoints and
    face centers. No smoothing is applied — vertices stay at exact midpoints.
    This is equivalent to USD's subdivisionScheme="bilinear".

    Args:
        points (np.array): Vertex positions, shape (N, D) where D is 2 or 3
        face_vertex_counts (np.array): Vertices per face (must all be 4 for quads)
        face_vertex_indices (np.array): Flat vertex index list
        levels (int): Number of subdivision levels to apply (default: 1, max: 6)
        face_uv_ranges (np.array, optional): Per-face UV ranges, shape (F, 2, 2)
            where each entry is [[u_min, v_min], [u_max, v_max]].
            If None, initialized to [[0,0],[1,1]] for each face.

    Returns:
        dict: Subdivision result with keys:
            - 'points': np.array (M, D) — refined vertex positions
            - 'face_vertex_counts': np.array (4*F, ) — all 4s
            - 'face_vertex_indices': np.array — refined topology
            - 'face_uv_ranges': np.array (4*F, 2, 2) — UV range per sub-face
            - 'face_quads': np.array (4*F, 4, D) — vertex positions for each sub-face
              in order [bottom-left, bottom-right, top-left, top-right]
    """
    if levels > 6:
        raise ValueError("Maximum 6 subdivision levels supported (4096 faces per original quad)")

    points = np.asarray(points, dtype=np.float64)
    face_vertex_counts = np.asarray(face_vertex_counts, dtype=np.int32)
    face_vertex_indices = np.asarray(face_vertex_indices, dtype=np.int32)

    if not np.all(face_vertex_counts == 4):
        raise ValueError("Bilinear subdivision requires all-quad input (all face_vertex_counts must be 4)")

    num_faces = len(face_vertex_counts)

    if face_uv_ranges is None:
        face_uv_ranges = np.zeros((num_faces, 2, 2), dtype=np.float64)
        face_uv_ranges[:, 1, :] = 1.0  # [[0,0],[1,1]] for each face

    for _ in range(levels):
        points, face_vertex_counts, face_vertex_indices, face_uv_ranges = \
            _bilinear_subdivide_once(points, face_vertex_counts, face_vertex_indices, face_uv_ranges)

    # Build face_quads: the 4 vertex positions for each face
    num_faces = len(face_vertex_counts)
    dim = points.shape[1]
    face_quads = np.zeros((num_faces, 4, dim), dtype=np.float64)
    for f in range(num_faces):
        idx_start = f * 4
        for v in range(4):
            face_quads[f, v] = points[face_vertex_indices[idx_start + v]]

    return {
        'points': points,
        'face_vertex_counts': face_vertex_counts,
        'face_vertex_indices': face_vertex_indices,
        'face_uv_ranges': face_uv_ranges,
        'face_quads': face_quads,
    }


def _bilinear_subdivide_once(points, face_vertex_counts, face_vertex_indices, face_uv_ranges):
    """
    Perform one level of bilinear subdivision.

    For each quad (v0, v1, v2, v3) with vertex ordering:
        v2---v3          v2--e23--v3
        |    |    =>     |  2 | 3  |
        |    |           e02--fc--e13
        v0---v1          |  0 | 1  |
                         v0--e01--v1

    Where:
        v0 = bottom-left  (u=0, v=0)
        v1 = bottom-right (u=1, v=0)
        v2 = top-left     (u=0, v=1)
        v3 = top-right    (u=1, v=1)
        e01 = midpoint(v0, v1), e23 = midpoint(v2, v3)
        e02 = midpoint(v0, v2), e13 = midpoint(v1, v3)
        fc  = face center = average(v0, v1, v2, v3)

    Sub-faces (in same vertex order [bl, br, tl, tr]):
        sub0: v0, e01, e02, fc     (uv: [u_min, v_min] to [u_mid, v_mid])
        sub1: e01, v1, fc, e13     (uv: [u_mid, v_min] to [u_max, v_mid])
        sub2: e02, fc, v2, e23     (uv: [u_min, v_mid] to [u_mid, v_max])
        sub3: fc, e13, e23, v3     (uv: [u_mid, v_mid] to [u_max, v_max])
    """
    num_faces = len(face_vertex_counts)
    dim = points.shape[1]

    # Build edge map: (min_idx, max_idx) -> edge midpoint index
    edge_midpoints = {}
    new_points_list = list(points)

    def get_edge_midpoint(a, b):
        key = (min(a, b), max(a, b))
        if key not in edge_midpoints:
            mid = (points[a] + points[b]) * 0.5
            edge_midpoints[key] = len(new_points_list)
            new_points_list.append(mid)
        return edge_midpoints[key]

    face_centers = []
    for f in range(num_faces):
        idx_start = f * 4
        verts = face_vertex_indices[idx_start:idx_start + 4]
        center = np.mean(points[verts], axis=0)
        fc_idx = len(new_points_list)
        new_points_list.append(center)
        face_centers.append(fc_idx)

    new_face_indices = []
    new_face_uv_ranges = []

    for f in range(num_faces):
        idx_start = f * 4
        v0, v1, v2, v3 = face_vertex_indices[idx_start:idx_start + 4]

        e01 = get_edge_midpoint(v0, v1)
        e13 = get_edge_midpoint(v1, v3)
        e23 = get_edge_midpoint(v2, v3)
        e02 = get_edge_midpoint(v0, v2)
        fc = face_centers[f]

        # UV range for this face
        uv_min = face_uv_ranges[f, 0]
        uv_max = face_uv_ranges[f, 1]
        u_mid = (uv_min[0] + uv_max[0]) * 0.5
        v_mid = (uv_min[1] + uv_max[1]) * 0.5

        # Sub-face 0: bottom-left
        new_face_indices.extend([v0, e01, e02, fc])
        new_face_uv_ranges.append([[uv_min[0], uv_min[1]], [u_mid, v_mid]])

        # Sub-face 1: bottom-right
        new_face_indices.extend([e01, v1, fc, e13])
        new_face_uv_ranges.append([[u_mid, uv_min[1]], [uv_max[0], v_mid]])

        # Sub-face 2: top-left
        new_face_indices.extend([e02, fc, v2, e23])
        new_face_uv_ranges.append([[uv_min[0], v_mid], [u_mid, uv_max[1]]])

        # Sub-face 3: top-right
        new_face_indices.extend([fc, e13, e23, v3])
        new_face_uv_ranges.append([[u_mid, v_mid], [uv_max[0], uv_max[1]]])

    new_points = np.array(new_points_list, dtype=np.float64)
    new_face_vertex_counts = np.full(num_faces * 4, 4, dtype=np.int32)
    new_face_vertex_indices = np.array(new_face_indices, dtype=np.int32)
    new_face_uv_ranges = np.array(new_face_uv_ranges, dtype=np.float64)

    return new_points, new_face_vertex_counts, new_face_vertex_indices, new_face_uv_ranges
