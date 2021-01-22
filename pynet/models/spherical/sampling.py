# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that provides spherical sampling & associated utilities.
"""

# Imports
import collections
import numpy as np
from math import sqrt, degrees
from sklearn.neighbors import BallTree
import networkx as nx


def interpolate(vertices, target_vertices, target_triangles):
    """ Interpolate missing data.

    Parameters
    ----------
    vertices: array (n_samples, n_dim)
        points of data set.
    target_vertices: array (n_query, n_dim)
        points to find interpolated texture for.
    target_triangles: array (n_query, 3)
        the mesh geometry definition.

    Returns
    -------
    interp_textures: array (n_query, n_feats)
        the interplatedd textures.
    """
    interp_textures = collections.OrderedDict()
    graph = vertex_adjacency_graph(target_vertices, target_triangles)
    common_vertices = downsample(target_vertices, vertices)
    missing_vertices = set(range(len(target_vertices))) - set(common_vertices)
    for node in sorted(graph.nodes):
        if node in common_vertices:
            interp_textures[node] = [node] * 2
        else:
            node_neighs = [idx for idx in graph.neighbors(node)
                           if idx in common_vertices]
            node_weights = np.linalg.norm(
                target_vertices[node_neighs] - target_vertices[node], axis=1)
            interp_textures[node] = node_neighs
    return interp_textures


def neighbors(vertices, triangles, depth=1, direct_neighbor=False):
    """ Build mesh vertices neighbors.

    Parameters
    ----------
    vertices: array (N, 3)
        the icosahedron vertices.
    triangles: array (N, 3)
        the icosahedron triangles.
    depth: int, default 1
        depth to stop the neighbors search, only paths of length <= depth are
        returned.
    direct_neighbor: bool, default False
        each spherical surface is composed of two types of vertices: 1) 12
        vertices with each having only 5 direct neighbors; and 2) the
        remaining vertices with each having 6 direct neighbors. For those
        vertices with 6 neighbors, DiNe assigns the index 1 to the center
        vertex and the indices 2–7 to its neighbors sequentially according
        to the angle between the vector of center vertex to neighboring vertex
        and the x-axis in the tangent plane. For the 12 vertices with only
        5 neighbors, DiNe assigns the indices both 1 and 2 to the center
        vertex, and indices 3–7 to the neighbors in the same way as those
        vertices with 6 neighbors.

    Returns
    --------
    neighs: dict
        a dictionary with vertices row index as keys and a dictionary of
        neighbors vertices row indexes organized by rungs as values.
    """
    graph = vertex_adjacency_graph(vertices, triangles)
    neighs = collections.OrderedDict()
    for node in sorted(graph.nodes):
        node_neighs = {}
        # node_neighs = [idx for idx in graph.neighbors(node)]
        for neigh, ring in nx.single_source_shortest_path_length(
                graph, node, cutoff=depth).items():
            if ring == 0:
                continue
            node_neighs.setdefault(ring, []).append(neigh)
        if direct_neighbor:
            _node_neighs = []
            if depth == 1:
                delta = np.pi / 4
            elif depth == 2:
                delta = np.pi / 8
            else:
                raise ValueError("Direct neighbors implemented only for "
                                 "depth <= 2.")
            for ring, ring_neighs in node_neighs.items():
                angles = np.asarray([
                    get_angle_with_xaxis(vertices[node], vertices[node], vec)
                    for vec in vertices[ring_neighs]])
                angles += delta
                angles = np.degrees(np.mod(angles, 2 * np.pi))
                ring_neighs = [x for _, x in sorted(
                    zip(angles, ring_neighs), key=lambda pair: pair[0])]
                if depth == 1 and ring == 1:
                    if len(ring_neighs) == 5:
                        ring_neighs.append(node)
                    elif len(ring_neighs) != 6:
                        raise ValueError("Mesh is not an icosahedron.")
                if depth == 2 and ring == 2:
                    ring_neighs = ring_neighs[1::2]
                    if len(_node_neighs) + len(ring_neighs) == 10:
                        ring_neighs.extend([node] * 2)
                    elif len(_node_neighs) + len(ring_neighs) == 11:
                        ring_neighs.append(node)
                    elif len(_node_neighs) + len(ring_neighs) != 12:
                        raise ValueError("Mesh is not an icosahedron.")
                _node_neighs.extend(ring_neighs)
            _node_neighs.append(node)
            node_neighs = _node_neighs
        neighs[node] = node_neighs

    return neighs


def vertex_adjacency_graph(vertices, triangles):
    """ Build a networkx graph representation of the vertices and
    their connections in the mesh.

    Parameters
    ----------
    vertices: array (N, 3)
        the icosahedron vertices.
    triangles: array (N, 3)
        the icosahedron triangles.

    Returns
    -------
    graph: networkx.Graph
        Graph representing vertices and edges between
        them where vertices are nodes and edges are edges

    Examples
    ----------
    This is useful for getting nearby vertices for a given vertex,
    potentially for some simple smoothing techniques.
    >>> graph = mesh.vertex_adjacency_graph
    >>> graph.neighbors(0)
    > [1, 3, 4]
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(vertices)))
    edges, edges_triangle = triangles_to_edges(triangles)
    edges_cache = []
    for idx1, idx2 in edges:
        smaller_index = min(idx1, idx2)
        greater_index = max(idx1, idx2)
        key = "{0}-{1}".format(smaller_index, greater_index)
        if key in edges_cache:
            continue
        edges_cache.append(key)
        graph.add_edge(smaller_index, greater_index)
    return graph


def get_angle_with_xaxis(center, normal, point):
    """ Project a point to the sphere tangent plane and compute the angle
    with the x-axis.

    Parameters
    ----------
    center: array (3, )
        a point in the plane.
    normal: array (3, )
        the normal to the plane.
    points: array (3, )
        the points to be projected.
    """
    # Assert is array
    center = np.asarray(center)
    normal = np.asarray(normal)
    point = np.asarray(point)

    # Project points to plane
    vector = point - center
    dist = np.dot(vector, normal)
    projection = point - normal * dist

    # Compute normal of the new projected x-axis and y-axis
    if center[0] != 0 or center[1] != 0:
        nx = np.cross(np.array([0, 0, 1]), center)
        ny = np.cross(center, nx)
    else:
        nx = np.array([1, 0, 0])
        ny = np.array([0, 1, 0])

    # Compute the angle between projected points and the x-axis
    vector = projection - center
    unit_vector = vector
    if np.linalg.norm(vector) != 0:
        unit_vector = unit_vector / np.linalg.norm(vector)
    unit_nx = nx / np.linalg.norm(nx)
    cos_theta = np.dot(unit_vector, unit_nx)
    if cos_theta > 1.:
        cos_theta = 1.
    elif cos_theta < -1.:
        cos_theta = -1.
    angle = np.arccos(cos_theta)
    if np.dot(unit_vector, ny) < 0:
        angle = 2 * np.pi - angle

    return angle


def triangles_to_edges(triangles, return_index=False):
    """ Given a list of triangles, return a list of edges.

    Parameters
    ----------
    triangles: array int (N, 3)
        Vertex indices representing triangles.

    Returns
    -------
    edges: array int (N * 3, 2)
        Vertex indices representing edges.
    triangles_index: array (N * 3, )
        Triangle indexes.
    """
    # Each triangles has three edges
    edges = triangles[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2))

    # Edges are in order of triangles due to reshape
    triangles_index = np.tile(
        np.arange(len(triangles)), (3, 1)).T.reshape(-1)

    return edges, triangles_index


def downsample(vertices, target_vertices):
    """ Downsample by finding nearest neighbors.

    Parameters
    ----------
    vertices: array (n_samples, n_dim)
        points of data set.
    target_vertices: array (n_query, n_dim)
        points to find nearest neighbors for.

    Returns
    -------
    nearest_idx: array (n_query, )
        index of nearest neighbor in target_vertices for every point in
        vertices.
    """
    if vertices.size == 0 or target_vertices.size == 0:
        return np.array([], int), np.array([])
    tree = BallTree(vertices, leaf_size=2)
    distances, nearest_idx = tree.query(
        target_vertices, return_distance=True, k=1)
    n_duplicates = len(nearest_idx) - len(np.unique(nearest_idx))
    if n_duplicates:
        raise RuntimeError("Could not downsample proprely, '{0}' duplicates "
                           "were found. Are you using an icosahedron "
                           "mesh?".format(n_duplicates))
    return nearest_idx.squeeze()


def neighbors_rec(vertices, triangles, size=5, zoom=5):
    """ Build rectangular grid neighbors and weights.

    Parameters
    ----------
    vertices: array (N, 3)
        the icosahedron vertices.
    triangles: array (N, 3)
        the icosahedron triangles.
    size: int, default 5
        the rectangular grid size.
    zoom: int, default 5
        scale factor applied on the unit sphere to control the neighborhood
        density.

    Returns
    --------
    neighs: array (N, size**2, 3)
        grid samples neighbors for each vertex.
    weights: array (N, size**2, 3)
        grid samples weights with neighbors for each vertex.
    grid_in_sphere: array (N, size**2, 3)
        zoomed rectangular grid on the sphere vertices.
    """
    grid_in_sphere = np.zeros((len(vertices), size**2, 3), dtype=float)
    neighs = np.zeros((len(vertices), size**2, 3), dtype=int)
    weights = np.zeros((len(vertices), size**2, 3), dtype=float)
    for idx1, node in enumerate(vertices):
        grid_in_sphere[idx1], _ = get_rectangular_projection(
            node, size=size, zoom=zoom)
        for idx2, point in enumerate(grid_in_sphere[idx1]):
            dist = np.linalg.norm(vertices - point, axis=1)
            ordered_neighs = np.argsort(dist)
            neighs[idx1, idx2] = ordered_neighs[:3]
            weights[idx1, idx2] = dist[neighs[idx1, idx2]]
    return neighs, weights, grid_in_sphere


def get_rectangular_projection(node, size=5, zoom=5):
    """ Project rectangular grid in 2D sapce into 3D spherical space.

    Parameters
    ----------
    node: array (3, )
        a point in the sphere.
    size: int, default 5
        the rectangular grid size.
    zoom: int, default 5
        scale factor applied on the unit sphere to control the neighborhood
        density.

    Returns
    -------
    grid_in_sphere: array (size**2, 3)
        zoomed rectangular grid on the sphere.
    grid_in_tplane: array (size**2, 3)
        zoomed rectangular grid in the tangent space.
    """
    # Check kernel size
    if (size % 2) == 0:
        raise ValueError("An odd kernel size is expected.")
    midsize = size // 2

    # Compute normal of the new projected x-axis and y-axis
    node = node.copy() * zoom
    if node[0] != 0 or node[1] != 0:
        nx = np.cross(np.array([0, 0, 1]), node)
        ny = np.cross(node, nx)
    else:
        nx = np.array([1, 0, 0])
        ny = np.array([0, 1, 0])
    nx = nx / np.linalg.norm(nx)

    ny = ny / np.linalg.norm(ny)

    # Caculate the grid coordinate in tangent plane and project back on sphere
    grid_in_tplane = np.zeros((size ** 2, 3))
    grid_in_sphere = np.zeros((size ** 2, 3))
    corner = node - midsize * nx + midsize * ny
    for row in range(size):
        for column in range(size):
            point = corner - row * ny + column * nx
            grid_in_tplane[row * size + column, :] = point
            grid_in_sphere[row * size + column, :] = (
                point / np.linalg.norm(point) * zoom)

    return grid_in_sphere, grid_in_tplane


def icosahedron(order=3):
    """ Define an icosahedron mesh of any order.

    Parameters
    ----------
    order: int, default 3
        the icosahedron order.

    Returns
    -------
    vertices: array (N, 3)
        the icosahedron vertices.
    triangles: array (N, 3)
        the icosahedron triangles.
    """
    middle_point_cache = {}
    r = (1 + np.sqrt(5)) / 2
    vertices = [
        normalize([-1, r, 0]),
        normalize([1, r, 0]),
        normalize([-1, -r, 0]),
        normalize([1, -r, 0]),
        normalize([0, -1, r]),
        normalize([0, 1, r]),
        normalize([0, -1, -r]),
        normalize([0, 1, -r]),
        normalize([r, 0, -1]),
        normalize([r, 0, 1]),
        normalize([-r, 0, -1]),
        normalize([-r, 0, 1])]
    triangles = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1]]

    for idx in range(order):
        subdiv = []
        for tri in triangles:
            v1 = middle_point(tri[0], tri[1], vertices, middle_point_cache)
            v2 = middle_point(tri[1], tri[2], vertices, middle_point_cache)
            v3 = middle_point(tri[2], tri[0], vertices, middle_point_cache)
            subdiv.append([tri[0], v1, v3])
            subdiv.append([tri[1], v2, v1])
            subdiv.append([tri[2], v3, v2])
            subdiv.append([v1, v2, v3])
        triangles = subdiv

    return np.asarray(vertices), np.asarray(triangles)


def normalize(vertex):
    """ Return vertex coordinates fixed to the unit sphere.
    """
    x, y, z = vertex
    length = sqrt(x**2 + y**2 + z**2)
    return [idx / length for idx in (x, y, z)]


def middle_point(point_1, point_2, vertices, middle_point_cache):
    """ Find a middle point and project to the unit sphere.
    """
    # We check if we have already cut this edge first to avoid duplicated verts
    smaller_index = min(point_1, point_2)
    greater_index = max(point_1, point_2)
    key = "{0}-{1}".format(smaller_index, greater_index)
    if key in middle_point_cache:
        return middle_point_cache[key]

    # If it's not in cache, then we can cut it
    vert_1 = vertices[point_1]
    vert_2 = vertices[point_2]
    middle = [sum(elems) / 2. for elems in zip(vert_1, vert_2)]
    vertices.append(normalize(middle))
    index = len(vertices) - 1
    middle_point_cache[key] = index

    return index


def number_of_ico_vertices(order=3):
    """ Get the number of vertices of an icosahedron of specific order.

    Parameters
    ----------
    order: int, default 3
        the icosahedron order.

    Returns
    -------
    vertices: array (N, 3)
        the icosahedron vertices.
    triangles: array (N, 3)
        the icosahedron triangles.
    """
    return 10 * 4 ** order + 2


if __name__ == "__main__":

    from pprint import pprint

    for order in range(5):
        print("=" * 5)
        print("ico", order)
        vertices, triangles = icosahedron(order=order)
        print("verts", vertices.shape, "tris", triangles.shape)
        print("n_verts", number_of_ico_vertices(order=order))

    print("=" * 5)
    vertices, triangles = icosahedron(order=1)
    print("verts", vertices.shape, "tris", triangles.shape)
    neighs = neighbors(vertices, triangles, depth=1, direct_neighbor=True)
    print("n_neighs", len(neighs))
    print("size_neighs", [len(elem) for elem in neighs.values()])
    print("neighs", neighs)

    print("=" * 5)
    target_vertices, _ = icosahedron(order=0)
    down_indexes = downsample(vertices, target_vertices)
    print("down 1 -> 0", down_indexes.shape)

    print("=" * 5)
    interp = interpolate(target_vertices, vertices, triangles)
    print("interp 0 -> 1")
    pprint(interp)

    print("=" * 5)
    neighs, weights, grid_in_sphere = neighbors_rec(
        vertices, triangles, size=5, zoom=5)
    print("rec_neighs", neighs.shape, "rec_weights", weights.shape,
          "rec_grid_shpere", grid_in_sphere.shape)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d

    vertices, triangles = icosahedron(order=2)
    neighs = neighbors(vertices, triangles, depth=2, direct_neighbor=True)
    colors = ["red", "green", "blue", "orange", "purple", "brown", "pink",
              "gray", "olive", "cyan", "yellow", "tan", "salmon", "violet",
              "steelblue", "lime", "navy"] * 5
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    fig, ax = plt.subplots(1, 1, subplot_kw={
        "projection": "3d", "aspect": "equal"}, figsize=(10, 10))
    ax.plot_trisurf(x, y, z, triangles=triangles, alpha=0., edgecolor="black",
                    facecolors="None", linewidth=0.1)
    for vidx in (0, 4):  # 13, 42, 0, 4
        for cnt, idx in enumerate(neighs[vidx]):
            point = vertices[idx]
            ax.scatter(point[0], point[1], point[2], marker="o", c=colors[cnt],
                       s=100)

    zoom = 5
    fig, ax = plt.subplots(1, 1, subplot_kw={
            "projection": "3d", "aspect": "equal"}, figsize=(10, 10))
    ax.plot_trisurf(x * zoom, y * zoom, z * zoom, triangles=triangles,
                    alpha=0., edgecolor="black", facecolors="None",
                    linewidth=0.2)
    for idx in (13, 40):
        node = vertices[idx]
        node_rec_neighs, node_tplane_neighs = get_rectangular_projection(
            node, size=3, zoom=zoom)
        ax.scatter(node[0] * zoom, node[1] * zoom, node[2] * zoom, marker="^",
                   c=colors[0], s=100)
        for cnt, point in enumerate(node_tplane_neighs):
            ax.scatter(point[0], point[1], point[2], marker="o",
                       c=colors[cnt + 1], s=5)
        for cnt, point in enumerate(node_rec_neighs):
            ax.scatter(point[0], point[1], point[2], marker="o",
                       c=colors[cnt + 1], s=20)

    plt.show()
