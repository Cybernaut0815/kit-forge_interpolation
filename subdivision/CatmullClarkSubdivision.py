import numpy as np
from collections import defaultdict


def catmull_clark_subdivide(points, face_vertex_counts, face_vertex_indices, levels=1,
                            face_uv_ranges=None,
                            crease_indices=None, crease_lengths=None, crease_sharpnesses=None,
                            corner_indices=None, corner_sharpnesses=None,
                            interpolate_boundary='edgeAndCorner'):
    """
    Perform Catmull-Clark subdivision on a quad mesh.

    Implements the standard Catmull-Clark algorithm with support for creases
    and corners, matching OpenSubdiv / USD subdivision behavior. Each quad is
    split into 4 sub-quads with smoothed vertex positions.

    Args:
        points (np.array): Vertex positions, shape (N, D) where D is 2 or 3
        face_vertex_counts (np.array): Vertices per face (must all be 4 for quads)
        face_vertex_indices (np.array): Flat vertex index list
        levels (int): Number of subdivision levels (default: 1, max: 6)
        face_uv_ranges (np.array, optional): Per-face UV ranges, shape (F, 2, 2).
            If None, initialized to [[0,0],[1,1]] for each face.
        crease_indices (np.array, optional): Vertex indices for crease edges,
            packed sequentially per crease (e.g., [v0, v1, v2, v3] for 2 creases of length 2).
        crease_lengths (np.array, optional): Number of vertices per crease chain.
        crease_sharpnesses (np.array, optional): Sharpness value per crease.
            Infinite sharpness makes a perfectly hard edge.
        corner_indices (np.array, optional): Vertex indices for sharp corners.
        corner_sharpnesses (np.array, optional): Sharpness per corner vertex.
        interpolate_boundary (str): Boundary interpolation rule:
            'none', 'edgeOnly', or 'edgeAndCorner' (default).

    Returns:
        dict: Same structure as bilinear_subdivide:
            - 'points', 'face_vertex_counts', 'face_vertex_indices',
              'face_uv_ranges', 'face_quads'
            Plus:
            - 'crease_indices', 'crease_lengths', 'crease_sharpnesses'
              (updated for the refined level)
    """
    if levels > 6:
        raise ValueError("Maximum 6 subdivision levels supported")

    points = np.asarray(points, dtype=np.float64)
    face_vertex_counts = np.asarray(face_vertex_counts, dtype=np.int32)
    face_vertex_indices = np.asarray(face_vertex_indices, dtype=np.int32)

    if not np.all(face_vertex_counts == 4):
        raise ValueError("Catmull-Clark subdivision requires all-quad input")

    num_faces = len(face_vertex_counts)
    if face_uv_ranges is None:
        face_uv_ranges = np.zeros((num_faces, 2, 2), dtype=np.float64)
        face_uv_ranges[:, 1, :] = 1.0

    for _ in range(levels):
        points, face_vertex_counts, face_vertex_indices, face_uv_ranges, \
            crease_indices, crease_lengths, crease_sharpnesses = \
            _catmull_clark_subdivide_once(
                points, face_vertex_counts, face_vertex_indices, face_uv_ranges,
                crease_indices, crease_lengths, crease_sharpnesses,
                corner_indices, corner_sharpnesses,
                interpolate_boundary)
        # Corners don't change index after first level in this simplified model
        # In practice, corner vertices keep their index, so we pass them through
        # For subsequent levels, corner sharpness is decremented
        if corner_sharpnesses is not None:
            corner_sharpnesses = np.maximum(corner_sharpnesses - 1.0, 0.0)

    # Build face_quads
    num_faces = len(face_vertex_counts)
    dim = points.shape[1]
    face_quads = np.zeros((num_faces, 4, dim), dtype=np.float64)
    for f in range(num_faces):
        idx_start = f * 4
        for v in range(4):
            face_quads[f, v] = points[face_vertex_indices[idx_start + v]]

    result = {
        'points': points,
        'face_vertex_counts': face_vertex_counts,
        'face_vertex_indices': face_vertex_indices,
        'face_uv_ranges': face_uv_ranges,
        'face_quads': face_quads,
    }
    if crease_indices is not None:
        result['crease_indices'] = crease_indices
        result['crease_lengths'] = crease_lengths
        result['crease_sharpnesses'] = crease_sharpnesses
    return result


def _quad_edges(v0, v1, v2, v3):
    """
    Return the 4 edge keys for a quad with layout [BL, BR, TL, TR].

    Edges: bottom(v0-v1), right(v1-v3), top(v2-v3), left(v0-v2).
    """
    return [
        (min(v0, v1), max(v0, v1)),
        (min(v1, v3), max(v1, v3)),
        (min(v2, v3), max(v2, v3)),
        (min(v0, v2), max(v0, v2)),
    ]


def _build_adjacency(num_points, face_vertex_counts, face_vertex_indices):
    """Build adjacency data structures for the mesh.

    Uses quad-specific edge connectivity: for a quad [v0, v1, v2, v3]
    with layout [BL, BR, TL, TR], the 4 edges are v0-v1, v1-v3, v2-v3, v0-v2.
    """
    num_faces = len(face_vertex_counts)

    # vertex_faces: which faces each vertex belongs to
    vertex_faces = defaultdict(list)
    # edge_faces: which faces each edge belongs to
    edge_faces = defaultdict(list)
    # vertex_edges: which edges each vertex belongs to
    vertex_edges = defaultdict(set)

    idx = 0
    for f in range(num_faces):
        n = face_vertex_counts[f]
        face_verts = face_vertex_indices[idx:idx + n]
        v0, v1, v2, v3 = int(face_verts[0]), int(face_verts[1]), int(face_verts[2]), int(face_verts[3])
        for v in [v0, v1, v2, v3]:
            vertex_faces[v].append(f)
        for edge_key in _quad_edges(v0, v1, v2, v3):
            edge_faces[edge_key].append(f)
            vertex_edges[edge_key[0]].add(edge_key)
            vertex_edges[edge_key[1]].add(edge_key)
        idx += n

    return vertex_faces, edge_faces, vertex_edges


def _build_crease_map(crease_indices, crease_lengths, crease_sharpnesses):
    """Build a map from edge key -> sharpness for crease edges."""
    crease_map = {}
    if crease_indices is None or crease_lengths is None or crease_sharpnesses is None:
        return crease_map

    idx = 0
    for c, length in enumerate(crease_lengths):
        sharpness = crease_sharpnesses[c]
        for i in range(int(length) - 1):
            v0 = int(crease_indices[idx + i])
            v1 = int(crease_indices[idx + i + 1])
            edge_key = (min(v0, v1), max(v0, v1))
            crease_map[edge_key] = sharpness
        idx += int(length)

    return crease_map


def _build_corner_map(corner_indices, corner_sharpnesses):
    """Build a map from vertex index -> sharpness for corner vertices."""
    corner_map = {}
    if corner_indices is None or corner_sharpnesses is None:
        return corner_map
    for i, v in enumerate(corner_indices):
        corner_map[int(v)] = float(corner_sharpnesses[i])
    return corner_map


def _catmull_clark_subdivide_once(points, face_vertex_counts, face_vertex_indices,
                                   face_uv_ranges,
                                   crease_indices, crease_lengths, crease_sharpnesses,
                                   corner_indices, corner_sharpnesses,
                                   interpolate_boundary):
    """Perform one level of Catmull-Clark subdivision."""
    num_points = len(points)
    num_faces = len(face_vertex_counts)
    dim = points.shape[1]

    vertex_faces, edge_faces, vertex_edges = _build_adjacency(
        num_points, face_vertex_counts, face_vertex_indices)
    crease_map = _build_crease_map(crease_indices, crease_lengths, crease_sharpnesses)
    corner_map = _build_corner_map(corner_indices, corner_sharpnesses)

    # ---- Step 1: Compute face points (average of face vertices) ----
    face_points = np.zeros((num_faces, dim), dtype=np.float64)
    face_verts_list = []
    idx = 0
    for f in range(num_faces):
        n = face_vertex_counts[f]
        verts = face_vertex_indices[idx:idx + n]
        face_verts_list.append(verts)
        face_points[f] = np.mean(points[verts], axis=0)
        idx += n

    # ---- Step 2: Compute edge points ----
    edge_point_map = {}  # edge_key -> (new_point_position, new_index)
    all_edges = list(edge_faces.keys())

    for edge_key in all_edges:
        v0, v1 = edge_key
        adj_faces = edge_faces[edge_key]
        is_boundary = len(adj_faces) == 1
        sharpness = crease_map.get(edge_key, 0.0)

        if is_boundary or sharpness >= 1.0:
            # Boundary or sharp crease: edge point = midpoint
            edge_pt = (points[v0] + points[v1]) * 0.5
        elif sharpness > 0.0:
            # Semi-sharp crease: blend between midpoint and smooth
            midpoint = (points[v0] + points[v1]) * 0.5
            smooth_pt = (points[v0] + points[v1] +
                         sum(face_points[f] for f in adj_faces)) / (2 + len(adj_faces))
            edge_pt = midpoint * sharpness + smooth_pt * (1.0 - sharpness)
        else:
            # Smooth edge: average of endpoints + adjacent face points
            edge_pt = (points[v0] + points[v1] +
                       sum(face_points[f] for f in adj_faces)) / (2 + len(adj_faces))

        edge_point_map[edge_key] = edge_pt

    # ---- Step 3: Compute new vertex positions ----
    new_vertex_positions = np.zeros((num_points, dim), dtype=np.float64)

    for v in range(num_points):
        adj_face_list = vertex_faces.get(v, [])
        adj_edge_list = vertex_edges.get(v, set())
        n = len(adj_face_list)

        if n == 0:
            new_vertex_positions[v] = points[v]
            continue

        # Check if vertex is a corner
        corner_sharpness_val = corner_map.get(v, 0.0)

        # Count sharp/crease edges at this vertex
        sharp_edges = []
        for ek in adj_edge_list:
            s = crease_map.get(ek, 0.0)
            is_boundary_edge = len(edge_faces.get(ek, [])) == 1
            if is_boundary_edge or s > 0.0:
                sharp_edges.append((ek, max(s, 10.0) if is_boundary_edge else s))

        is_boundary_vertex = any(len(edge_faces.get(ek, [])) == 1 for ek in adj_edge_list)
        num_sharp = len(sharp_edges)

        if corner_sharpness_val >= 1.0 or num_sharp > 2:
            # Corner vertex: stays fixed
            new_vertex_positions[v] = points[v]

        elif num_sharp == 2 and not is_boundary_vertex:
            # Crease vertex (interior with exactly 2 sharp edges)
            # Use crease rule: 1/8 * (e0 + e1) + 6/8 * v
            avg_sharpness = np.mean([s for _, s in sharp_edges])
            e_verts = []
            for ek, _ in sharp_edges:
                other = ek[1] if ek[0] == v else ek[0]
                e_verts.append(points[other])

            crease_pt = (e_verts[0] + e_verts[1]) / 8.0 + points[v] * 6.0 / 8.0

            if avg_sharpness >= 1.0:
                new_vertex_positions[v] = crease_pt
            else:
                # Blend with smooth rule
                F_avg = np.mean([face_points[f] for f in adj_face_list], axis=0)
                edge_mids = []
                for ek in adj_edge_list:
                    other = ek[1] if ek[0] == v else ek[0]
                    edge_mids.append((points[v] + points[other]) * 0.5)
                R_avg = np.mean(edge_mids, axis=0) * 2.0
                smooth_pt = (F_avg + R_avg + points[v] * (n - 3)) / n
                new_vertex_positions[v] = crease_pt * avg_sharpness + smooth_pt * (1.0 - avg_sharpness)

        elif is_boundary_vertex:
            # Boundary vertex: average of adjacent boundary edge midpoints
            if interpolate_boundary == 'none':
                new_vertex_positions[v] = points[v]
            else:
                boundary_neighbors = []
                for ek in adj_edge_list:
                    if len(edge_faces.get(ek, [])) == 1:
                        other = ek[1] if ek[0] == v else ek[0]
                        boundary_neighbors.append(points[other])
                if len(boundary_neighbors) >= 2:
                    # Boundary crease rule: 1/8 * (n0 + n1) + 6/8 * v
                    new_vertex_positions[v] = (boundary_neighbors[0] + boundary_neighbors[1]) / 8.0 + points[v] * 6.0 / 8.0
                elif len(boundary_neighbors) == 1:
                    # Boundary corner: stay fixed
                    new_vertex_positions[v] = points[v]
                else:
                    new_vertex_positions[v] = points[v]

                # Corner of boundary: a vertex with total valence <= 2
                # (only 2 incident edges) is a mesh corner and should be pinned.
                if interpolate_boundary == 'edgeAndCorner' and len(adj_edge_list) <= 2:
                    new_vertex_positions[v] = points[v]

        else:
            # Smooth interior vertex: standard Catmull-Clark rule
            # new_v = (F + 2R + (n-3)*v) / n
            # F = average of adjacent face points
            # R = average of adjacent edge midpoints
            F_avg = np.mean([face_points[f] for f in adj_face_list], axis=0)
            edge_mids = []
            for ek in adj_edge_list:
                other = ek[1] if ek[0] == v else ek[0]
                edge_mids.append((points[v] + points[other]) * 0.5)
            R_avg = np.mean(edge_mids, axis=0)
            new_vertex_positions[v] = (F_avg + 2.0 * R_avg + (n - 3) * points[v]) / n

    # ---- Step 4: Assemble new topology ----
    # New points: [updated original vertices, face points, edge points]
    # Index mapping
    new_points_list = list(new_vertex_positions)

    # Face point indices
    face_point_indices = {}
    for f in range(num_faces):
        face_point_indices[f] = len(new_points_list)
        new_points_list.append(face_points[f])

    # Edge point indices
    edge_point_indices = {}
    for edge_key in all_edges:
        edge_point_indices[edge_key] = len(new_points_list)
        new_points_list.append(edge_point_map[edge_key])

    # Build new faces — same topology split as bilinear
    new_face_indices = []
    new_face_uv_ranges = []

    for f in range(num_faces):
        verts = face_verts_list[f]
        v0, v1, v2, v3 = verts[0], verts[1], verts[2], verts[3]

        e01_key = (min(v0, v1), max(v0, v1))
        e13_key = (min(v1, v3), max(v1, v3))
        e23_key = (min(v2, v3), max(v2, v3))
        e02_key = (min(v0, v2), max(v0, v2))

        e01 = edge_point_indices[e01_key]
        e13 = edge_point_indices[e13_key]
        e23 = edge_point_indices[e23_key]
        e02 = edge_point_indices[e02_key]
        fc = face_point_indices[f]

        uv_min = face_uv_ranges[f, 0]
        uv_max = face_uv_ranges[f, 1]
        u_mid = (uv_min[0] + uv_max[0]) * 0.5
        v_mid = (uv_min[1] + uv_max[1]) * 0.5

        # Sub-face 0: bottom-left [v0, e01, e02, fc]
        new_face_indices.extend([v0, e01, e02, fc])
        new_face_uv_ranges.append([[uv_min[0], uv_min[1]], [u_mid, v_mid]])

        # Sub-face 1: bottom-right [e01, v1, fc, e13]
        new_face_indices.extend([e01, v1, fc, e13])
        new_face_uv_ranges.append([[u_mid, uv_min[1]], [uv_max[0], v_mid]])

        # Sub-face 2: top-left [e02, fc, v2, e23]
        new_face_indices.extend([e02, fc, v2, e23])
        new_face_uv_ranges.append([[uv_min[0], v_mid], [u_mid, uv_max[1]]])

        # Sub-face 3: top-right [fc, e13, e23, v3]
        new_face_indices.extend([fc, e13, e23, v3])
        new_face_uv_ranges.append([[u_mid, v_mid], [uv_max[0], uv_max[1]]])

    new_points = np.array(new_points_list, dtype=np.float64)
    new_fvc = np.full(num_faces * 4, 4, dtype=np.int32)
    new_fvi = np.array(new_face_indices, dtype=np.int32)
    new_face_uv_ranges = np.array(new_face_uv_ranges, dtype=np.float64)

    # ---- Step 5: Update crease sharpnesses for next level ----
    # Each crease edge is split into 2 sub-edges. Sharpness decremented by 1.
    new_crease_indices_list = []
    new_crease_lengths_list = []
    new_crease_sharpnesses_list = []

    if crease_indices is not None and crease_lengths is not None and crease_sharpnesses is not None:
        c_idx = 0
        for c, length in enumerate(crease_lengths):
            old_sharpness = float(crease_sharpnesses[c])
            new_sharpness = max(old_sharpness - 1.0, 0.0)
            if new_sharpness > 0.0:
                for i in range(int(length) - 1):
                    v0 = int(crease_indices[c_idx + i])
                    v1 = int(crease_indices[c_idx + i + 1])
                    edge_key = (min(v0, v1), max(v0, v1))
                    ep = edge_point_indices[edge_key]
                    # Original edge v0-v1 splits into v0-ep and ep-v1
                    new_crease_indices_list.extend([v0, ep])
                    new_crease_lengths_list.append(2)
                    new_crease_sharpnesses_list.append(new_sharpness)
                    new_crease_indices_list.extend([ep, v1])
                    new_crease_lengths_list.append(2)
                    new_crease_sharpnesses_list.append(new_sharpness)
            c_idx += int(length)

    new_crease_indices = np.array(new_crease_indices_list, dtype=np.int32) if new_crease_indices_list else None
    new_crease_lengths = np.array(new_crease_lengths_list, dtype=np.int32) if new_crease_lengths_list else None
    new_crease_sharpnesses = np.array(new_crease_sharpnesses_list, dtype=np.float64) if new_crease_sharpnesses_list else None

    return (new_points, new_fvc, new_fvi, new_face_uv_ranges,
            new_crease_indices, new_crease_lengths, new_crease_sharpnesses)
