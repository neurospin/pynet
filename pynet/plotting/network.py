# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that provides tools to display a graph.
"""

# System import
import sys
import os
import time
from pprint import pprint
import tempfile
import weakref
import operator
import tempfile

# Third party import
import torch
import hiddenlayer as hl
from PySide2 import QtCore, QtGui, QtWidgets
from torchviz import make_dot

# Module import
from .graph import Graph, GraphNode
from pynet.plotting.colors import *


def plot_net_rescue(model, shape, outfileroot=None):
    """ Save a PNG file containing the network graph representation.

    Parameters
    ----------
    model: Net
        the network model.
    shape: list of int
        the shape of a classical input batch dataset.
    outfileroot: str, default None
        the file path without extension.

    Returns
    -------
    outfile: str
        the path to the generated PNG.
    """
    x = torch.randn(shape)
    graph = make_dot(model(x), params=dict(model.named_parameters()))
    graph.format = "png"
    if outfileroot is None:
        dirpath = tempfile.mkdtemp()
        basename = "pynet_graph"
    else:
        dirpath = os.path.dirname(outfileroot)
        basename = os.path.basename(outfileroot)
    graph.render(directory=dirpath, filename=basename, view=True)
    return os.path.join(dirpath, basename + ".png")


def plot_net(model, shape, static=True, outfileroot=None):
    """ Save a PDF file containing the network graph representation.

    Sometimes the 'get_trace_graph' pytorch function fails: use the
    'plot_net_rescue' function insteed.

    Parameters
    ----------
    model: Net
        the network model.
    shape: list of int
        the shape of a classical input batch dataset.
    static: bool, default True
        create a static or dynamic view.
    outfileroot: str, default None
        the file path without extension to generate PDF.

    Returns
    -------
    outfile: str
        the path to the generated PDF.
    """
    # Create application
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    # Create view
    hl_graph = hl.build_graph(model, torch.zeros(shape))
    hl_graph.theme = hl.graph.THEMES["blue"].copy()
    outfile = None
    if outfileroot is not None:
        hl_graph.save(outfileroot)
        outfile = outfileroot + ".pdf"
        if not os.path.isfile(outfile):
            raise ValueError("'{0}' has not been generated.".format(outfile))
    if static:
        def draw(widget, surface):
            page.render(surface)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfileroot = os.path.join(tmpdir, "graph")
            hl_graph.save(tmpfileroot, format="png")
            tmpfile = tmpfileroot + ".png"
            widget = PDFView(tmpfile)
            view = QtWidgets.QScrollArea()
            view.setWidgetResizable(True)
            view.setWidget(widget)
    else:
        graph = Graph()
        nodes_map = {}
        cnt = 1
        for key, node in hl_graph.nodes.items():
            label = node.title
            if node.caption:
                label += node.caption
            if node.repeat:
                label += str(node.repeat)
            nodes_map[key] = "{0}-{1}".format(cnt, label)
            cnt += 1
        for key, node in hl_graph.nodes.items():
            graph.add_node(GraphNode(str(nodes_map[key]), node))
        for key1, key2, label in hl_graph.edges:
            if isinstance(label, (list, tuple)):
                label = "x".join([str(l or "?") for l in label])
            graph.add_link(str(nodes_map[key1]), str(nodes_map[key2]))
        view = GraphView(graph)

    # Display
    view.show()
    app.exec_()

    return outfile


class PDFView(QtWidgets.QWidget):
    """ A widget to visualize a PDF graph.
    """
    def __init__(self, path):
        """ Initialize the PDFView class
        """
        super(PDFView, self).__init__()
        self.path = path
        layout = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel()
        layout.addWidget(self.label)
        self.pixmap = QtGui.QPixmap(self.path)
        self.label.setPixmap(self.pixmap)


class Control(QtWidgets.QGraphicsPolygonItem):
    """ Create a glyph for each control connection.
    """

    def __init__(self, name, height, width, optional, parent=None):
        """ Initilaize the Control class.

        Parameters
        ----------
        name: str
            the control name.
        height, width: int
            the control size.
        optional: bool
            option to color the glyph.
        """
        # Inheritance
        super(Control, self).__init__(parent)

        # Class parameters
        self.name = name
        self.optional = optional
        color = self._color(optional)
        self.brush = QtGui.QBrush(QtCore.Qt.SolidPattern)
        self.brush.setColor(color)

        # Set graphic item properties
        self.setAcceptedMouseButtons(QtCore.Qt.LeftButton)

        # Define the widget
        polygon = QtGui.QPolygonF([
            QtCore.QPointF(0, 0), QtCore.QPointF(width, (height - 5) / 2.0),
            QtCore.QPointF(0, height - 5)])
        self.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        self.setPolygon(polygon)
        self.setBrush(self.brush)
        self.setZValue(3)

    def _color(self, optional):
        """ Define the color of a control glyph depending on its status.

        Parameters
        ----------
        optional: bool (mandatory)
            option to color the glyph.

        Returns
        -------
        color: QColor
            the glyph color.
        """
        if optional:
            color = QtCore.Qt.darkGreen
        else:
            color = QtCore.Qt.black
        return color

    def get_control_point(self):
        """ Give the relative location of the control glyph in the parent
        widget.

        Returns
        -------
        position: QPointF
            the control glyph position.
        """
        point = QtCore.QPointF(
            self.boundingRect().size().width() / 2.0,
            self.boundingRect().size().height() / 2.0)
        return self.mapToParent(point)


class Node(QtWidgets.QGraphicsItem):
    """ A box node.
    """
    _colors = {
        "default": (RED_1, RED_2, LIGHT_RED_1, LIGHT_RED_2),
        "choice1": (SAND_1, SAND_2, LIGHT_SAND_1, LIGHT_SAND_2),
        "choice2": (DEEP_PURPLE_1, DEEP_PURPLE_2, PURPLE_1, PURPLE_2),
        "choice3": (BLUE_1, BLUE_2, LIGHT_BLUE_1, LIGHT_BLUE_2)
    }

    def __init__(self, name, inputs, outputs, active=True, style=None,
                 graph=None, parent=None):
        """ Initilaize the Node class.

        Parameters
        ----------
        name: string
            a name for the box node.
        inputs: list of str
            the box input controls. If None no input will be created.
        outputs: list of str
            the box output controls. If None no output will be created.
        active: bool, default True)
            a special color will be applied on the node rendering depending
            of this parameter.
        style: string, default None
            the style that will be applied to tune the box rendering.
        graph: Graph, default None
            a sub-graph item.
        """
        # Inheritance
        super(Node, self).__init__(parent)

        # Class parameters
        self.style = style or "default"
        self.name = name
        self.graph = graph
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.active = active
        self.input_controls = {}
        self.output_controls = {}
        self.embedded_box = None

        # Set graphic item properties
        self.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
        self.setAcceptedMouseButtons(
            QtCore.Qt.LeftButton | QtCore.Qt.RightButton |
            QtCore.Qt.MiddleButton)

        # Define rendering colors
        bgd_color_indices = [2, 3]
        if self.active:
            bgd_color_indices = [0, 1]
        self.background_brush = self._get_brush(
            *operator.itemgetter(*bgd_color_indices)(self._colors[self.style]))
        self.title_brush = self._get_brush(
            *operator.itemgetter(2, 3)(self._colors[self.style]))

        # Construct the node
        self._build()

    def get_title(self):
        """ Create a title for the node.
        """
        return self.name

    def _build(self, margin=5):
        """ Create a node reprensenting a box.

        Parameters
        ----------
        margin: int (optional, default 5)
            the default margin.
        """
        # Create a title for the node
        self.title = QtGui.QGraphicsTextItem(self.get_title(), self)
        font = self.title.font()
        font.setWeight(QtGui.QFont.Bold)
        self.title.setFont(font)
        self.title.setPos(margin, margin)
        self.title.setZValue(2)
        self.title.setParentItem(self)

        # Define the default control position
        control_position = (
            margin + margin + self.title.boundingRect().size().height())

        # Create the input controls
        for input_name in self.inputs:

            # Create the control representation
            control_glyph, control_text = self._create_control(
                input_name, control_position, is_output=False, margin=margin)

            # Update the class parameters
            self.input_controls[input_name] = (control_glyph, control_text)

            # Update the next control position
            control_position += control_text.boundingRect().size().height()

        # Create the output controls
        for output_name in self.outputs:

            # Create the control representation
            control_glyph, control_text = self._create_control(
                output_name, control_position, is_output=True, margin=margin)

            # Update the class parameters
            self.output_controls[output_name] = (control_glyph, control_text)

            # Update the next control position
            control_position += control_text.boundingRect().size().height()

        # Define the box node
        self.box = QtGui.QGraphicsRectItem(self)
        self.box.setBrush(self.background_brush)
        self.box.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        self.box.setZValue(-1)
        self.box.setParentItem(self)
        self.box.setRect(self.contentsRect())
        self.box_title = QtGui.QGraphicsRectItem(self)
        rect = self.title.mapRectToParent(self.title.boundingRect())
        brect = self.contentsRect()
        brect.setWidth(brect.right() - margin)
        rect.setWidth(brect.width())
        self.box_title.setRect(rect)
        self.box_title.setBrush(self.title_brush)
        self.box_title.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        self.box_title.setParentItem(self)

    def _create_control(self, control_name, control_position, is_output=False,
                        control_width=12, margin=5):
        """ Create a control representation: small glyph and control name.

        Parameters
        ----------
        control_name: str (mandatory)
            the name of the control to render.
        control_position: int (mandatory)
            the position (height) of the control to render.
        control_name: bool (optional, default False)
            an input control glyph is diplayed on the left while an output
            control glyph is displayed on the right.
        control_width: int (optional, default 12)
            the default size of the control glyph.
        margin: int (optional, default 5)
            the default margin.

        Returns
        -------
        control_text: QGraphicsTextItem
            the control text item.
        control_glyph: Control
            the associated control glyph item.
        """
        # Detect if the control is optional
        is_optional = False

        # Create the control representation
        control_text = QtGui.QGraphicsTextItem(self)
        control_text.setHtml(control_name)
        control_name = "{0}:{1}".format(self.name, control_name)
        control_glyph = Control(
            control_name, control_text.boundingRect().size().height(),
            control_width, optional=is_optional, parent=self)
        control_text.setZValue(2)
        control_glyph_width = control_glyph.boundingRect().size().width()
        control_title_width = self.title.boundingRect().size().width()
        control_text.setPos(control_glyph_width + margin, control_position)
        if is_output:
            control_glyph.setPos(
                control_title_width - control_glyph_width,
                control_position)
        else:
            control_glyph.setPos(margin, control_position)
        control_text.setParentItem(self)
        control_glyph.setParentItem(self)

        return control_glyph, control_text

    def _get_brush(self, color1, color2):
        """ Create a brush that has a style, a color, a gradient and a texture.

        Parameters
        ----------
        color1, color2: QtGui.QColor (mandatory)
            edge box colors used to define the gradient.
        """
        gradient = QtGui.QLinearGradient(0, 0, 0, 50)
        gradient.setColorAt(0, color1)
        gradient.setColorAt(1, color2)
        return QtGui.QBrush(gradient)

    def contentsRect(self):
        """ Returns the area inside the widget's margins.

        Returns
        -------
        brect: QRectF
            the bounding rectangle (left, top, right, bottom).
        """
        first = True
        excluded = []
        for name in ("box", "box_title"):
            if hasattr(self, name):
                excluded.append(getattr(self, name))
        for child in self.childItems():
            if not child.isVisible() or child in excluded:
                continue
            item_rect = self.mapRectFromItem(child, child.boundingRect())
            if first:
                first = False
                brect = item_rect
            else:
                if item_rect.left() < brect.left():
                    brect.setLeft(item_rect.left())
                if item_rect.top() < brect.top():
                    brect.setTop(item_rect.top())
                if item_rect.right() > brect.right():
                    brect.setRight(item_rect.right())
                if item_rect.bottom() > brect.bottom():
                    brect.setBottom(item_rect.bottom())
        return brect

    def boundingRect(self):
        """ Returns the bounding rectangle of the given text as it will appear
        when drawn inside the rectangle beginning at the point (x , y ) with
        width w and height h.

        Returns
        -------
        brect: QRectF
            the bounding rectangle (x, y, w, h).
        """
        brect = self.contentsRect()
        brect.setRight(brect.right())
        brect.setBottom(brect.bottom())
        return brect

    def paint(self, painter, option, widget=None):
        pass

    def mouseDoubleClickEvent(self, event):
        """ If a sub-graph is available emit a 'subgraph_clicked' signal.
        """
        if self.graph is not None:
            self.scene().subgraph_clicked.emit(self.name, self.graph,
                                               event.modifiers())
            event.accept()
        else:
            event.ignore()

    def add_subgraph_view(self, graph, margin=5):
        """ Display the a sub-graph box in a node.

        Parameters
        ----------
        graph: Graph
            the sub-graph box to display.
        """
        # Create a embedded proxy view
        if self.embedded_box is None:
            view = GraphView(graph)
            proxy_view = EmbeddedSubGraphItem(view)
            view._graphics_item = weakref.proxy(proxy_view)
            proxy_view.setParentItem(self)
            posx = margin + self.box.boundingRect().width()
            proxy_view.setPos(posx, margin)
            self.embedded_box = proxy_view

        # Change visibility property of the embedded proxy view
        else:
            if self.embedded_box.isVisible():
                self.embedded_box.hide()
            else:
                self.embedded_box.show()


class EmbeddedSubGraphItem(QtWidgets.QGraphicsProxyWidget):
    """ QGraphicsItem containing a sub-graph box view.
    """

    def __init__(self, sub_graph_view):
        """ Initialize the EmbeddedSubGraphItem.

        Parameters
        ----------
        sub_graph_view: GraphView
            the sub-graph view.
        """
        # Inheritance
        super(EmbeddedSubGraphItem, self).__init__()

        # Define rendering options
        sub_graph_view.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOn)
        sub_graph_view.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOn)
        # sub_graph_view.setFixedSize(400, 600)

        # Add the sub-graph widget
        self.setWidget(sub_graph_view)

        # sub_graph_view.setSizePolicy(
        #     QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        # self.setSizePolicy(
        #     QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)


class Link(QtWidgets.QGraphicsPathItem):
    """ A link between boxes.
    """

    def __init__(self, src_position, dest_position, parent=None):
        """ Initilaize the Link class.

        Parameters
        ----------
        src_position: QPointF (mandatory)
            the source control glyph position.
        dest_position: QPointF (mandatory)
            the destination control glyph position.
        """
        # Inheritance
        super(Link, self).__init__(parent)

        # Define the color rendering
        pen = QtGui.QPen()
        pen.setWidth(2)
        pen.setBrush(RED_2)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        pen.setJoinStyle(QtCore.Qt.RoundJoin)
        self.setPen(pen)

        # Draw the link
        path = QtGui.QPainterPath()
        path.moveTo(src_position.x(), src_position.y())
        path.cubicTo(src_position.x() + 100, src_position.y(),
                     dest_position.x() - 100, dest_position.y(),
                     dest_position.x(), dest_position.y())
        self.setPath(path)
        self.setZValue(0.5)

    def update(self, src_position, dest_position):
        """ Update the link extreme positions.

        Parameters
        ----------
        src_position: QPointF (mandatory)
            the source control glyph position.
        dest_position: QPointF (mandatory)
            the destination control glyph position.
        """
        path = QtGui.QPainterPath()
        path.moveTo(src_position.x(), src_position.y())
        path.cubicTo(src_position.x() + 100, src_position.y(),
                     dest_position.x() - 100, dest_position.y(),
                     dest_position.x(), dest_position.y())
        self.setPath(path)


class GraphScene(QtWidgets.QGraphicsScene):
    """ Define a scene representing a graph.
    """
    # Signal emitted when a sub graph has to be open
    subgraph_clicked = QtCore.Signal(str, Graph, QtCore.Qt.KeyboardModifiers)

    def __init__(self, graph, parent=None):
        """ Initilaize the GraphScene class.

        Parameters
        ----------
        graph: Graph
            graph to be displayed.
        parent: QWidget, default None)
            parent widget.
        """
        # Inheritance
        super(GraphScene, self).__init__(parent)

        # Class parameters
        self.graph = graph
        self.gnodes = {}
        self.glinks = {}
        self.gpositions = {}

        # Add event to upadate links
        self.changed.connect(self.update_links)

    def update_links(self):
        """ Update the node positions and associated links.
        """
        for node in self.items():
            if isinstance(node, Node):
                self.gpositions[node.name] = node.pos()

        for linkdesc, link in self.glinks.items():
            # Parse the link description
            src_control, dest_control = self.parse_link_description(linkdesc)

            # Get the source and destination nodes/controls
            src_gnode = self.gnodes[src_control[0]]
            dest_gnode = self.gnodes[dest_control[0]]
            src_gcontrol = src_control[1]
            dest_gcontrol = dest_control[1]

            # Update the current link
            src_control_glyph = src_gnode.output_controls[src_gcontrol][0]
            dest_control_glyph = dest_gnode.input_controls[dest_gcontrol][0]
            link.update(
                src_gnode.mapToScene(src_control_glyph.get_control_point()),
                dest_gnode.mapToScene(dest_control_glyph.get_control_point()))

    def draw(self):
        """ Draw the scene representing the graph.
        """
        # Add the graph graph
        for box_name, box in self.graph._nodes.items():

            # Define the box type and check if we are dealing with a graph
            # box
            if isinstance(box.meta, Graph):
                style = "choice1"
            else:
                style = "choice3"

            # Add the box
            self.add_box(
                box_name,
                inputs=[""] * (0 if box.links_from_degree == 0 else 1),
                outputs=[""] * (0 if box.links_to_degree == 0 else 1),
                active=True,
                style=style,
                graph=box.meta)

        # If no node position is defined used an automatic setup
        # based on a graph representation
        if self.gpositions == {}:
            scale = 0.0
            for node in self.gnodes.values():
                scale = max(node.box.boundingRect().width(), scale)
                scale = max(node.box.boundingRect().height(), scale)
            scale *= 4
            box_positions = self.graph.layout(scale=scale)
            for node_name, node_pos in box_positions.items():
                self.gnodes[node_name].setPos(QtCore.QPointF(*node_pos))

        # Create the links between the boxes
        for from_box_name, to_box_name in self.graph._links:
            self.add_link("{0}.->{1}.".format(from_box_name, to_box_name))

    def parse_link_description(self, linkdesc):
        """ Parse a link description.

        Parameters
        ----------
        linkdesc: string (mandatory)
            link representation with the source and destination separated
            by '->' and control desriptions of the form
            '<box_name>.<control_name>' or '<control_name>' for graph
            input or output controls.

        Returns
        -------
        src_control: 2-uplet
            the source control representation (box_name, control_name).
        dest_control: 2-uplet
            the destination control representation (box_name, control_name).
        """
        # Parse description

        srcdesc, destdesc = linkdesc.split("->")
        src_control = srcdesc.split(".")
        dest_control = destdesc.split(".")

        # Deal with graph input and output controls
        if len(src_control) == 1:
            src_control.insert(0, "inputs")
        if len(dest_control) == 1:
            dest_control.insert(0, "outputs")

        return tuple(src_control), tuple(dest_control)

    def add_box(self, name, inputs, outputs, active=True, style=None,
                graph=None):
        """ Add a box in the graph representation.

        Parameters
        ----------
        name: string
            a name for the box.
        inputs: list of str
            the box input controls.
        outputs: list of str
            the box output controls.
        active: bool, default True
            a special color will be applied on the box rendering depending
            of this parameter.
        style: string, default None
            the style that will be applied to tune the box rendering.
        graph: Graph, default None
            the sub-graph item.
        """
        # Create the node widget that represents the box
        box_node = Node(name, inputs, outputs, active=active, style=style,
                        graph=graph)

        # Update the scene
        self.addItem(box_node)
        node_position = self.gpositions.get(name)
        if node_position is not None:
            box_node.setPos(node_position)
        self.gnodes[name] = box_node

    def add_link(self, linkdesc):
        """ Define a link between two nodes in the graph.

        Parameters
        ----------
        linkdesc: string (mandatory)
            link representation with the source and destination separated
            by '->' and control desriptions of the form
            '<box_name>.<control_name>' or '<control_name>' for graph
            input or output controls.
        """
        # Parse the link description
        src_control, dest_control = self.parse_link_description(linkdesc)

        # Get the source and destination nodes/controls
        src_gnode = self.gnodes[src_control[0]]
        dest_gnode = self.gnodes[dest_control[0]]
        src_gcontrol = src_control[1]
        dest_gcontrol = dest_control[1]

        # Create the link
        src_control_glyph = src_gnode.output_controls[src_gcontrol][0]
        dest_control_glyph = dest_gnode.input_controls[dest_gcontrol][0]
        glink = Link(
            src_gnode.mapToScene(src_control_glyph.get_control_point()),
            dest_gnode.mapToScene(dest_control_glyph.get_control_point()))

        # Update the scene
        self.addItem(glink)
        self.glinks[linkdesc] = glink

    def keyPressEvent(self, event):
        """ Display the graph box positions when the 'p' key is pressed.
        """
        super(GraphScene, self).keyPressEvent(event)
        if not event.isAccepted() and event.key() == QtCore.Qt.Key_P:
            event.accept()
            posdict = dict([(key, (value.x(), value.y()))
                            for key, value in self.gpositions.items()])
            pprint(posdict)

    def helpEvent(self, event):
        """ Display tooltips on controls and links.
        """
        item = self.itemAt(event.scenePos())
        if isinstance(item, Control):
            item.setToolTip("type: {0} - optional: {1}".format(
                item.control.__class__.__name__, item.optional))
        super(GraphScene, self).helpEvent(event)


class GraphView(QtWidgets.QGraphicsView):
    """ Graph representation (using boxes and arrows).

    Based on Qt QGraphicsView, this can be used as a Qt QWidget.

    Qt signals are emitted:

    * on a double click on a sub-graph box to display the sub-graph. If
      'ctrl' is pressed a new window is created otherwise the view is
      embedded.
    * on the wheel to zoom in or zoom out.
    * on the kewboard 'p' key to display the box node positions.

    Attributes
    ----------
    scene: GraphScene
        the main scene.
    """
    # Signal emitted when a sub graph has to be open
    subgraph_clicked = QtCore.Signal(str, Graph, QtCore.Qt.KeyboardModifiers)

    def __init__(self, graph, parent=None):
        """ Initilaize the GraphView class.

        Parameters
        ----------
        graph: Graph
            graph to be displayed.
        parent: QWidget, default None
            parent widget.
        """
        # Inheritance
        super(GraphView, self).__init__(parent)

        # Class parameters
        self.scene = None

        # Check that we have a graph
        if not isinstance(graph, Graph):
            raise Exception("'{0}' is not a valid graph.".format(graph))

        # Create the graph representing.
        self.set_graph(graph)

    def set_graph(self, graph):
        """ Assigns a new graph to the view.

        Parameters
        ----------
        graph: Graph
            graph to be displayed.
        """
        # Define the graph box positions
        if hasattr(graph, "_box_positions"):
            box_positions = dict(
                (box_name, QtCore.QPointF(*box_position))
                for box_name, box_position in graph._box_positions.items())
        else:
            box_positions = {}

        # Create the scene
        self.scene = GraphScene(graph, self)
        self.scene.gpositions = box_positions
        self.scene.draw()

        # Update the current view
        self.setWindowTitle("Graph representation")
        self.setScene(self.scene)

        # Try to initialize the current view scale factor
        if hasattr(graph, "_scale"):
            self.scale(graph.scale, graph.scale)

        # Define signals
        self.scene.subgraph_clicked.connect(self.subgraph_clicked)
        self.scene.subgraph_clicked.connect(self.display_subgraph)

    def zoom_in(self):
        """ Zoom the view in by applying a 1.2 zoom factor.
        """
        self.scale(1.2, 1.2)

    def zoom_out(self):
        """ Zoom the view out by applying a 1 / 1.2 zoom factor.
        """
        self.scale(1.0 / 1.2, 1.0 / 1.2)

    def display_subgraph(self, node_name, graph, modifiers):
        """ Event to display the selected sub-graph.

        If 'ctrl' is pressed the a new window is created, otherwise the new
        view will be embedded in its parent node box.

        Parameters
        ----------
        node_name: str
            the node name.
        graph: Graph
            the sub-graph box to display.
        """
        # Open a new window
        if modifiers & QtCore.Qt.ControlModifier:
            view = GraphView(graph)
            QtCore.QObject.setParent(view, self.window())
            view.setAttribute(QtCore.Qt.WA_DeleteOnClose)
            view.setWindowTitle(node_name)
            view.show()

        # Embedded sub-graph inside its parent node
        else:
            node = self.scene.gnodes.get(node_name)
            node.add_subgraph_view(graph)

    def wheelEvent(self, event):
        """ Change the scene zoom factor.
        """
        item = self.itemAt(event.pos())
        if not isinstance(item, QtGui.QGraphicsProxyWidget):
            if event.delta() < 0:
                self.zoom_out()
            else:
                self.zoom_in()
            event.accept()
        else:
            super(GraphView, self).wheelEvent(event)
