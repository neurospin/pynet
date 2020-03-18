# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Module that provides logging utilities.
"""


# System import
import os
import logging
import collections
import time
import datetime
import pickle

# Third party import
import numpy as np
from tabulate import tabulate


# Global parameters
logger = logging.getLogger("pynet")


class History(object):
    """ Track training progress by following the some metrics.
    """
    def __init__(self, name, verbose=0):
        """ Initilize the class.

        Parameters
        ----------
        name: str
            the object name.
        verbose: int, default 0
            control the verbosity level.
        """
        self.name = name
        self.verbose = verbose
        self.step = None
        self.metrics = set()
        self.history = collections.OrderedDict()

    def __repr__(self):
        """ Display the history.
        """
        table = []
        for step in self.steps:
            values = []
            for metric in self.metrics:
                values.append(self.history[step][metric])
            table.append([step] + values)
        return tabulate(table, headers=self.metrics)

    def log(self, step, **kwargs):
        """ Record some metrics at a specific step.

        Example:
            state = History()
            state.log(1, loss=1., accuracy=0.)

        If logging the same metrics for one specific step, new values
        overwrite older ones.

        Parameters
        ----------
        step: int or uplet
            The step name: we can use a tuple to log the fold, the epoch
            or the step within the epoch.
        kwargs
            The metrics to be logged.
        """
        if not isinstance(step, (int, tuple)):
            raise ValueError("Step must be an int or a tuple.")
        self.step = step
        self.metrics |= set(kwargs.keys())
        if step not in self.history:
            self.history[step] = {}
        for key, val in kwargs.items():
            self.history[step][key] = val
        self.history[step]["__timestamp__"] = time.time()

    @property
    def steps(self):
        """ Returns a list of all steps.
        """
        return list(self.history.keys())

    def __getitem__(self, metric):
        steps = self.steps
        data = np.array([self.history[step].get(metric) for step in steps])
        return steps, data

    def summary(self):
        last_step = self.steps[-1]
        msg = "{:6s} {:15s}".format(self.name, repr(last_step))
        for key in self.metrics:
            msg += "{:6s}:{:10f}  ".format(key, self.history[last_step][key])
        msg += "{}".format(str(self.get_total_time()))
        logger.info(msg)

    def get_total_time(self):
        """ Returns the total period between the first and last steps.
        """
        seconds = (
            self.history[self.steps[-1]]["__timestamp__"]
            - self.history[self.steps[0]]["__timestamp__"])
        return datetime.timedelta(seconds=seconds)

    def save(self, outdir, fold, epoch):
        outfile = os.path.join(
            outdir, "{0}_{1}_epoch_{2}.pkl".format(self.name, fold, epoch))
        with open(outfile, "wb") as open_file:
            pickle.dump(self, open_file)

    @classmethod
    def load(cls, file_name):
        with open(file_name, "rb") as open_file:
            return pickle.load(open_file)
