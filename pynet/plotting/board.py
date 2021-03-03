# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Common functions to display a dynamic board.
"""

# Import
import sys
import json
import logging
import numpy as np
import visdom
import torch
from subprocess import Popen, PIPE


# Global parameters
logger = logging.getLogger("pynet")


class Board(object):
    """ Define a dynamic board.

    It can be used to gather interesting plottings during a training.
    """
    def __init__(self, port=8097, host="http://localhost", env="main",
                 display_pred=False, prepare_pred=None):
        """ Initilaize the class.

        Parameters
        ----------
        port: int, default 8097
            the port on which the visdom server is launched.
        host: str, default 'http://localhost'
            the host on which visdom is launched.
        env: str, default 'main'
            the environment to be used.
        display_pred: bool, default False
            if set render the predicted images.
        prepare_pred: callable, defaultt None
            a function that transforms the predictions into a Nx1xXxY or
            Nx3xXxY array, with N the number of images.
        """
        self.port = port
        self.host = host
        self.env = env
        self.display_pred = display_pred
        self.prepare_pred = prepare_pred
        self.plots = {}
        logger.debug("Create viewer on host {0} port {1}.".format(host, port))
        self.viewer = visdom.Visdom(
            port=self.port, server=self.host, env=self.env)
        while len(logging.root.handlers) > 0:
            logging.root.removeHandler(logging.root.handlers[-1])
        self.server = None
        if not self.viewer.check_connection():
            self._create_visdom_server()
        current_data = json.loads(self.viewer.get_window_data())
        for key in current_data:
            logger.debug("Closing plot {0}.".format(key))
            self.viewer.close(win=key)

    def __del__(self):
        """ Class destructor.
        """
        if self.server is not None:
            logger.debug("Stoping visdom server.")
            self.server.kill()
            self.server.wait()

    def _create_visdom_server(self):
        """ It starts a new visdom server.
        """
        current_python = sys.executable
        cmd = "{0} -m visdom.server -p {1}".format(current_python, self.port)
        logger.debug("Starting visdom server:\n{0}".format(cmd))
        self.server = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def update_plots(self, data, epoch):
        """ Update/create plots from the input data.

        Parameters
        ----------
        data: dict
            the name and new value of the plot to be updated.
        epoch: int
            the current epoch.
        """
        logger.debug("Board data update:\n{0}".format(data))
        # current_data = json.loads(self.viewer.get_window_data())
        for key, val in data.items():
            if key == "val_pred":
                if not self.display_pred:
                    continue
                images = np.asarray(val)
                if self.prepare_pred is not None:
                    images = self.prepare_pred(val)
                if images.ndim != 4:
                    raise ValueError(
                        "You must define a function that transforms the "
                        "predictions into a Nx1xXxY or Nx3xXxY array, with N "
                        "the number of images.")
                self.viewer.images(
                    images,
                    opts={
                        "title": key,
                        "caption": "y_pred"},
                    win=key)
            else:
                self.viewer.line(
                    X=np.asarray([epoch]),
                    Y=np.asarray([val]),
                    opts={
                        "title": key,
                        "xlabel": "iterations",
                        "ylabel": key},
                    update="append",
                    win=key)


def update_board(signal):
    """ Callback to update visdom board visualizer.

    Parameters
    ----------
    signal: SignalObject
        an object with the trained model 'object', the emitted signal
        'signal', the epoch number 'epoch' and the fold index 'fold'.
    """
    net = signal.object.model
    emitted_signal = signal.signal
    epoch = signal.epoch
    fold = signal.fold
    board = signal.object.board
    data = {}
    for key in signal.keys:
        if key in ("epoch", "fold"):
            continue
        value = getattr(signal, key)
        if key == "scheduler" and value is None:
            continue
        if key == "scheduler" and value is not None:
            if hasattr(value, "get_last_lr"):
                value = value.get_last_lr()[0]
            else:
                value = value._last_lr[0]
            key += "_lr"
        if isinstance(value, torch.Tensor):
            value = value.cpu().detach().numpy().tolist()
        data[key] = value
    board.update_plots(data, epoch)
