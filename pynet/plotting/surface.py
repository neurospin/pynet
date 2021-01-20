# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that provides utility functions to display surfaces.
"""

# Imports
import numpy as np
from matplotlib import cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_trisurf(fig, ax, vertices, triangles, texture=None, vmin=None,
                 vmax=None):
    """ Display a tri surface.

    Parameters
    ----------
    fig: Figure
        the matplotlib figure.
    ax: Axes3D
        axis to display the surface plot.
    vertices: array (N, 3)
        the surface vertices.
    triangles: array (N, 3)
        the surface triangles.
    texture: array (N,), default None
        a texture to display on the surface.
    vmin: float, default None
        minimum value to map.
    vmax: float, default None
        maximum value to map.
    """

    # Parameters
    if vmin is None:
        vmin = texture.min()
    if vmax is None:
        vmax = texture.max()
    if texture is None:
        texture = np.ones((len(vertices), ))

    # Display tri surface
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    norm = colors.Normalize(vmin=0, vmax=vmax, clip=False)
    facecolors = cm.coolwarm(norm(texture))
    triangle_vertices = np.array([vertices[tri] for tri in triangles])
    polygon = Poly3DCollection(triangle_vertices, facecolors=facecolors,
                               edgecolors="black")
    ax.add_collection3d(polygon)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # Add colorbar
    m = cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)
    m.set_array(texture)
    fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)

    # Get rid of the panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
