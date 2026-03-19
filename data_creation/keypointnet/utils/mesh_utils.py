import numpy as np

from sklearn.neighbors import NearestNeighbors


def find_adjacent_faces(faces, face_idx):
    """
    Find adjacent faces of a specific face in a mesh using a numpy-optimized approach.

    :param faces: Faces of mesh (m x 3), where each row is a face and columns are indices of vertices
    :param face_idx: Index of the face to find adjacent faces for
    :return: List with indices of adjacent faces for the specified face_idx, -1 if an adjacent face is not present
    """
    num_faces = faces.shape[0]
    if face_idx < 0 or face_idx >= num_faces:
        raise IndexError("face_idx is out of bounds")

    # Flatten the array of faces to handle edges easily
    all_edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    # repeat each face index three times
    face_indices = np.repeat(np.arange(num_faces), 3)

    # Create a structured array to use numpy's advanced indexing and sorting
    dtype = [('edge', all_edges.dtype, (2,)), ('face', face_indices.dtype)]
    structured_edges = np.empty(all_edges.shape[0], dtype=dtype)
    structured_edges['edge'] = all_edges
    structured_edges['face'] = face_indices

    # Sort edges first by edge, then by face index
    structured_edges.sort(order=['edge', 'face'])

    # Find edges that appear exactly twice (i.e., they are between two faces)
    mask = np.all(structured_edges['edge'][1:] ==
                  structured_edges['edge'][:-1], axis=1)
    adjacency_pairs = np.vstack(
        (structured_edges['face'][1:][mask], structured_edges['face'][:-1][mask])).T

    # Initialize the list of adjacent faces with -1 (indicating no adjacent face)
    adjacent_faces = [-1, -1, -1]

    # Process only the adjacency pairs involving the face of interest
    for pair in adjacency_pairs:
        if face_idx in pair:
            other_face = pair[1] if pair[0] == face_idx else pair[0]
            for i in range(3):
                if np.all(faces[face_idx, [i, (i + 1) % 3]] == faces[other_face, [(j + 1) % 3, j]] for j in range(3)):  # type: ignore[call-overload, arg-type]
                    adjacent_faces[i] = other_face

    return adjacent_faces


def find_adjacent_faces_n_ring(faces, face_idx, n_ring=2):
    """
    Find n-ring adjacent faces of a specific face in a mesh using a numpy-optimized approach.

    :param faces: Faces of mesh (m x 3), where each row is a face and columns are indices of vertices
    :param face_idx: Index of the face to find adjacent faces for
    :param n_ring: The ring number to find adjacency (1 means direct neighbors, 2 means neighbors of neighbors, etc.)
    :return: List with indices of adjacent faces for the specified face_idx, up to the n-th ring
    """
    num_faces = faces.shape[0]
    if face_idx < 0 or face_idx >= num_faces:
        raise IndexError("face_idx is out of bounds")

    # Initialize a list of current ring faces starting with the given face
    current_ring_faces = [face_idx]
    # Initialize a set of all visited faces to avoid revisiting
    all_adjacent_faces = set()

    for _ in range(n_ring):
        next_ring_faces = []
        for current_face in current_ring_faces:
            # Flatten the array of faces to handle edges easily
            all_edges = np.sort(
                faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
            # repeat each face index three times
            face_indices = np.repeat(np.arange(num_faces), 3)

            # Create a structured array to use numpy's advanced indexing and sorting
            dtype = [('edge', all_edges.dtype, (2,)),
                     ('face', face_indices.dtype)]
            structured_edges = np.empty(all_edges.shape[0], dtype=dtype)
            structured_edges['edge'] = all_edges
            structured_edges['face'] = face_indices

            # Sort edges first by edge, then by face index
            structured_edges.sort(order=['edge', 'face'])

            # Find edges that appear exactly twice (i.e., they are between two faces)
            mask = np.all(structured_edges['edge'][1:]
                          == structured_edges['edge'][:-1], axis=1)
            adjacency_pairs = np.vstack(
                (structured_edges['face'][1:][mask], structured_edges['face'][:-1][mask])).T

            # Find the adjacent faces for the current face
            adjacent_faces = set(adjacency_pairs[np.any(
                adjacency_pairs == current_face, axis=1)].flatten())
            # remove the current face itself
            adjacent_faces.discard(current_face)

            next_ring_faces.extend(adjacent_faces)

        # Update the list of current ring faces and include them in all visited faces
        current_ring_faces = list(set(next_ring_faces) - all_adjacent_faces)
        all_adjacent_faces.update(current_ring_faces)

    return list(all_adjacent_faces)


def prepare_face_adjacency(faces):
    """
    Prepare and return an adjacency list for all faces in the mesh.

    :param faces: Faces of the mesh (m x 3), where each row is a face and columns are indices of vertices
    :return: A list of sets, where each set contains indices of directly adjacent faces
    """
    num_faces = faces.shape[0]
    # Flatten the array of faces to handle edges easily
    all_edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    face_indices = np.repeat(np.arange(num_faces), 3)

    dtype = [('edge', all_edges.dtype, (2,)), ('face', face_indices.dtype)]
    structured_edges = np.empty(all_edges.shape[0], dtype=dtype)
    structured_edges['edge'] = all_edges
    structured_edges['face'] = face_indices

    structured_edges.sort(order=['edge', 'face'])

    mask = np.all(structured_edges['edge'][1:] ==
                  structured_edges['edge'][:-1], axis=1)
    adjacency_pairs = np.vstack(
        (structured_edges['face'][1:][mask], structured_edges['face'][:-1][mask])).T

    adjacency_list = [set() for _ in range(num_faces)]
    for f1, f2 in adjacency_pairs:
        adjacency_list[f1].add(f2)
        adjacency_list[f2].add(f1)

    return adjacency_list


def find_n_ring_neighbors(adjacency_list, face_idx, n_ring):
    """
    Find n-ring neighbors of a specific face given a pre-computed adjacency list.

    :param adjacency_list: List of sets containing adjacent faces for each face in the mesh
    :param face_idx: Index of the face to find neighbors for
    :param n_ring: Number of rings to consider for neighborhood
    :return: Set of indices of adjacent faces up to the n-th ring
    """
    current_ring_faces = {face_idx}
    all_adjacent_faces = set()

    for _ in range(n_ring):
        next_ring_faces = set()
        for current_face in current_ring_faces:
            next_ring_faces.update(adjacency_list[current_face])

        next_ring_faces.difference_update(all_adjacent_faces)
        all_adjacent_faces.update(next_ring_faces)
        current_ring_faces = next_ring_faces

    return all_adjacent_faces


def find_all_n_ring_neighbors(adjacency_list, n_ring=1):
    """
    Compute n-ring neighbors for all faces using the pre-computed adjacency list.

    :param adjacency_list: List of sets containing adjacent faces for each face in the mesh
    :param n_ring: Number of rings to consider for neighborhood
    :return: A list of sets, each set contains indices of up to n-ring adjacent faces for each face
    """
    num_faces = len(adjacency_list)
    n_ring_neighbors_list = [set() for _ in range(num_faces)]

    for face_idx in range(num_faces):
        current_ring_faces = {face_idx}
        all_adjacent_faces = set(current_ring_faces)

        for _ in range(n_ring):
            next_ring_faces = set()
            for current_face in current_ring_faces:
                next_ring_faces.update(adjacency_list[current_face])

            next_ring_faces.difference_update(all_adjacent_faces)
            all_adjacent_faces.update(next_ring_faces)
            current_ring_faces = next_ring_faces

        n_ring_neighbors_list[face_idx] = all_adjacent_faces

    return n_ring_neighbors_list


def get_per_view_visible_pnts(mesh, face_neighbors, pnts, pnt_ids, pnt_face_ids, view_intersection_pnts, view_face_ids, thresh=0.02):
    if (len(pnts) == 0):
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    h, w = view_face_ids.shape[0:2]

    view_face_ids_set = set(list(np.unique(view_face_ids.flatten())))
    view_face_ids_set.remove(-1)

    ################################################################################################
    # First for each sampled point on the mesh, get its nearest point from the intersection points
    ################################################################################################
    # Cast their points to be very far
    view_intersection_pnts[view_face_ids == -
                           1] = np.array([1000000, 1000000, 1000000])
    view_intersection_pnts_flatten = view_intersection_pnts.reshape((h*w, 3))

    thresh *= np.ptp(mesh.bounding_box.bounds)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(
        view_intersection_pnts_flatten)
    distances, indices = nbrs.kneighbors(pnts)
    distances = distances.reshape(len(pnts))
    indices = indices.reshape(len(pnts))
    pnt_vis_mask = distances <= thresh

    pnts = pnts[pnt_vis_mask]
    pnt_face_ids = pnt_face_ids[pnt_vis_mask]
    pnt_ids = pnt_ids[pnt_vis_mask]
    distances = distances[pnt_vis_mask]
    indices = indices[pnt_vis_mask]

    ################################################################################################
    # Second for point which passed the first filter, remove the ones whose face is not visible
    ################################################################################################
    pnt_vis_mask = []
    for el in pnt_face_ids:
        faces_list = set(list(face_neighbors[el]) + [el])
        # faces_list = face_neighbors.get(el, find_adjacent_faces(mesh.faces, el)) + [el]
        # face_neighbors[el] = faces_list
        is_found = np.array(
            [f in view_face_ids_set for f in faces_list], dtype=bool)
        pnt_vis_mask.append(np.any(is_found))

    pnt_vis_mask = np.array(pnt_vis_mask, dtype=bool)

    pnts = pnts[pnt_vis_mask]
    pnt_face_ids = pnt_face_ids[pnt_vis_mask]
    pnt_ids = pnt_ids[pnt_vis_mask]
    indices = indices[pnt_vis_mask]
    distances = distances[pnt_vis_mask]

    ################################################################################################
    # Now get the pixel coordinates for the visible points
    ################################################################################################
    n = len(pnts)
    assert h == w
    rows = indices // w
    cols = indices % h
    pos = np.hstack([rows.reshape(n, 1), cols.reshape(n, 1)])
    

    return pnts, pnt_ids, pnt_face_ids, pos


def get_all_visible_pnt_feats(mesh, all_pnts, all_pnt_ids, all_pnt_face_ids, all_views_intersection_pnts, all_views_face_ids):
    n_views = len(all_views_face_ids)
    ret2 = []

    print(f"Computing face n_ring neighborhood...")
    adjacency_list = prepare_face_adjacency(mesh.faces)
    face_neighbors = find_all_n_ring_neighbors(adjacency_list)
    # face_neighbors = dict()
    print(f"Computing face n_ring=1 neighborhood. done.")

    for i in range(n_views):
        pnts, pnt_ids, pnt_face_ids, pos = get_per_view_visible_pnts(  # type: ignore[misc]
            mesh, face_neighbors, all_pnts, all_pnt_ids, all_pnt_face_ids, all_views_intersection_pnts[i], all_views_face_ids[i])
        ret2.append([pnts, pnt_ids, pnt_face_ids, pos])
    return ret2


def project_images_to_pointcloud(mesh, all_pnts, all_pnts_face_ids, all_views_intersection_pnts, all_views_face_ids):
    all_pnt_ids = np.arange(len(all_pnts))
    return get_all_visible_pnt_feats(mesh, all_pnts, all_pnt_ids, all_pnts_face_ids, all_views_intersection_pnts, all_views_face_ids)
