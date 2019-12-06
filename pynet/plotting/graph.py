# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


# Thrid party imports
import numpy


class GraphNode(object):
    """ Simple Graph Node Structure.

    Attributes
    ----------
    name : str
        the node name
    meta : object
        a python object stored in the node
    links_to : list
         object to store the graph edges: sucessor
    links_from : list
        object to store the graph edges: predecessor
    links_to_degree : int
        degree of the node regarding the successors
    links_from_degree : int
        degree of the node regarding the predecessors
    """
    def __init__(self, name, meta):
        """ Create a Graph Node.

        Parameters
        ----------
        name: str (mandatory)
            the name of the node.

        meta: object
            an python object to store in the node.
        """
        self.name = name
        self.meta = meta
        # variables to store the graph edges
        self.links_to = []
        self.links_from = []
        # the degree of the node
        self.links_to_degree = 0
        self.links_from_degree = 0

    def add_link_to(self, node):
        """ Method to add a Successor.

        Parameters
        ----------
        node: GraphNode (mandatory)
            the successor node.
        """
        if node not in self.links_to:
            self.links_to.append(node)
            self.links_to_degree += 1

    def remove_link_to(self, node):
        """ Method to remove a Successor.

        Parameters
        ----------
        node: GraphNode (mandatory)
            the successor node.
        """
        if node in self.links_to:
            self.links_to.remove(node)
            self.links_to_degree -= 1

    def add_link_from(self, node):
        """ Method to add a Predecessor.

        Parameters
        ----------
        node: GraphNode (mandatory)
            the predecessor node.
        """
        if node not in self.links_from:
            self.links_from.append(node)
            self.links_from_degree += 1

    def remove_link_from(self, node):
        """ Method to remove a Predecessor.

        Parameters
        ----------
        node: GraphNode (mandatory)
            the predecessor node.
        """
        if node in self.links_from:
            self.links_from.remove(node)
            self.links_from_degree -= 1


class Graph(object):
    """ Simple Graph Structure on which we want to perform a
    topological tree (no cycle).

    The algorithm is based on the R.E. Tarjanlinear linear
    optimization (O(N+A)).

    Attributes
    ----------
    _nodes : dict
        the graph nodes {node.name: node}
    _links : list
        graph edges (from_node, to_node)
    """
    def __init__(self):
        """ Create a Graph
        """
        self._nodes = {}
        self._links = []

    def add_node(self, node):
        """ Method to add a GraphNode in the Graph.

        Parameters
        ----------
        node: GraphNode (mandatory)
            the node to insert.
        """
        if not isinstance(node, GraphNode):
            raise Exception("'{0}' is not a GraphNode.".format(node))
        if node.name in self._nodes:
            raise ValueError("'{0}' is already a GraphNode name.".format(
                node.name))
        self._nodes[node.name] = node

    def add_graph(self, graph):
        """ Method to add a Graph in the Graph.

        Parameters
        ----------
        graph: Graph (mandatory)
            the graph to insert.
        """
        for node_name, node in graph._nodes.items():
            self.add_node(node)
        for link in graph._links:
            self.add_link(*link)

    def remove_node(self, node_name):
        """ Method to remove a GraphNode from the Graph.

        Parameters
        ----------
        node: string (mandatory)
            the name of the node to remove.
        """
        if node_name not in self._nodes:
            raise Exception("'{0}' is not a valid GraphNode name.".format(
                node_name))
        node = self._nodes[node_name]
        for to_node in node.links_to:
            to_node.remove_link_from(node)
        for from_node in node.links_from:
            from_node.remove_link_to(node)
        del self._nodes[node_name]

    def find_node(self, node_name):
        """ Method to find a GraphNode in the Graph.

        Parameters
        ----------
        node_name: str (mandatory)
            the name of the desired node.
        """
        if node_name in self._nodes:
            return self._nodes[node_name]
        return None

    def add_link(self, from_node, to_node):
        """ Method to add an edge between two GraphNodes of the Graph.

        Parameters
        ----------
        from_node: GraphNode (mandatory)
            node link representation of the form '<node>.<control>'.
        to_node: GraphNode (mandatory)
            node link representation of the form '<node>.<control>'.
        """
        if from_node not in self._nodes:
            raise Exception("Node '{0}' is not defined in the Graph.".format(
                from_node))
        if to_node not in self._nodes:
            raise Exception("Node '{0}' is not defined in the Graph.".format(
                to_node))
        if (from_node, to_node) not in self._links:
            self._nodes[to_node].add_link_from(self._nodes[from_node])
            self._nodes[from_node].add_link_to(self._nodes[to_node])
            self._links.append((from_node, to_node))

    def topological_sort(self):
        """ Perform the topological sort: find an order in which all the
        nodes can be taken.

        Step 1: Identify nodes that have no incoming link (nnil).
        Step 2: Loop until there are nnil
        a) Delete the current nodes c_nnil of in-degree 0.
        b) Place it in the output.
        c) Remove all its outgoing links from the graph.
        d) If the node has in-degree 0, add the node to nnil.
        Step 3: Assert that there is no loop in the graph.

        Returns
        -------
        output: list of tuple
            a list of ordered nodes with a tuple element containing the node
            name and the node meta element.
        """
        ordered_nodes = []

        # Step 1
        nnil = self.available_nodes()

        # Step 2
        while len(nnil):
            # -- a
            c_nnil = nnil.pop()
            # -- b
            ordered_nodes.append(c_nnil)
            # -- c
            for node in c_nnil.links_to:
                node.remove_link_from(c_nnil)
            # -- d
                if node.links_from_degree == 0:
                    nnil.append(node)

        # Step 3
        if len(ordered_nodes) == len(self._nodes):
            return [(node.name, node.meta) for node in ordered_nodes]
        else:
            raise Exception("There is a loop in the Graph.")

    def available_nodes(self):
        """ List the nodes that have no incoming link.
        """
        nnil = []
        for name, node in self._nodes.items():
            if node.links_from_degree == 0:
                nnil.append(node)
        return nnil

    def adjacency_matrix(self):
        """ Compute the graph adjacency matrix.

        Returns
        -------
        adjacency_matrix: array
            Graph adjacency matrix.
        node_names: list
            Ordered node names.
        """
        node_names = list(self._nodes.keys())
        nb_of_nodes = len(node_names)
        indices = dict(zip(node_names, range(nb_of_nodes)))
        adjacency_matrix = numpy.zeros((nb_of_nodes, nb_of_nodes),
                                       dtype=numpy.single)
        for node_name, node in self._nodes.items():
            for link_node in node.links_to:
                adjacency_matrix[indices[node.name],
                                 indices[link_node.name]] += 1
            for link_node in node.links_from:
                adjacency_matrix[indices[node.name],
                                 indices[link_node.name]] += 1
        adjacency_matrix = numpy.asmatrix(adjacency_matrix)
        return adjacency_matrix, node_names

    def layout(self, scale=1.0):
        """ Position nodes using the eigenvectors of the graph Laplacian.

        Parameters
        ----------
        scale: float (optional, default 1.0)
            Scale factor for positions. The nodes are positioned
            in a box of size [0, scale] x [0, scale].

        Returns
        -------
        pos: dict
           A dictionary of positions keyed by node.
        """
        # Function parameters
        dim = 2
        center = numpy.zeros(dim)
        pos = None

        # Compute symmetrize adjacency matrix
        adjacency_matrix, node_names = self.adjacency_matrix()
        adjacency_matrix += numpy.transpose(adjacency_matrix)

        # Default cases
        if len(self._nodes) == 0:
            return {}
        elif len(self._nodes) == 1:
            pos = numpy.array([center])
        elif len(self._nodes) == 2:
            pos = numpy.array([numpy.array([0.5, 0]), numpy.array([-0.5, 0])])

        # Compute the node position from the adjacency matrix
        else:
            pos = self._spectral(adjacency_matrix)

        # Rescale to (-scale,scale) in all axes
        # > shift origin to (0,0)
        max_coordinate = 0
        for i in range(dim):
            pos[:, i] -= pos[:, i].mean()
            max_coordinate = max(pos[:, i].max(), max_coordinate)
        # > rescale to (-scale,scale) in all directions preserving aspect
        if max_coordinate > 0:
            for i in range(dim):
                pos[:, i] *= scale / max_coordinate

        # Organize position
        pos = dict(zip(node_names, pos))

        return pos

    def _spectral(self, adjacency_matrix):
        """ Use dense eigenvalue solver.

        Parameters
        ----------
        adjacency_matrix: array (mandatory)
            Graph adjacency matrix.

        Returns
        -------
        pos: array
            the node positions.
        """
        # Function parameters
        dim = adjacency_matrix.ndim
        nb_of_nodes = adjacency_matrix.shape[0]
        adjacency_matrix = numpy.asarray(adjacency_matrix)

        # Form the Laplacian matrix
        I = numpy.identity(nb_of_nodes, dtype=adjacency_matrix.dtype)
        D = I * numpy.sum(adjacency_matrix, axis=1)  # diagonal of degrees
        L = D - adjacency_matrix

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = numpy.linalg.eig(L)

        # Sort and keep smallest nonzero: 0 index is zero eigenvalue
        index = numpy.argsort(eigenvalues)[1: dim + 1]

        return numpy.real(eigenvectors[:, index])
