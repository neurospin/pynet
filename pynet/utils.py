# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
A module with common functions.
"""

# System import
import shutil
import logging
import tempfile
import warnings
import os
import re
import sys
import inspect

# Third party imports
import torch
import numpy as np


ALLOWED_LAYERS = [
    torch.nn.Conv2d,
    torch.nn.Conv3d,
    torch.nn.ConvTranspose2d,
    torch.nn.ConvTranspose3d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.Linear
]
LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}
logger = logging.getLogger("pynet")


class TemporaryDirectory(object):
    """ Securely creates a temporary directory. The resulting object can be
    used as a context manager. When the returned object is used as a context
    manager, the name will be assigned to the target of the as clause in the
    with statement, if there is one.
    """
    def __init__(self, dir=None, prefix=None, name=None):
        """ Initialize the TempDir class.

        Parameters
        ----------
        dir: str, default None
            the location where the temporary folder is created. If specified
            the folder is persistent.
        prefix: str, default None
            if set the directory name will begin with that prefix.
        name: str, default
            if set the directory name will have this name.
        """
        self.tmpdir = None
        self.dir = dir
        self.prefix = prefix
        self.name = name
        self.delete = self.dir is None
        return

    def __enter__(self):
        if self.dir is not None and self.name is not None:
            self.tmpdir = os.path.join(self.dir, self.name)
            if not os.path.isdir(self.tmpdir):
                os.mkdir(self.tmpdir)
        else:
            self.tmpdir = tempfile.mkdtemp(dir=self.dir, prefix=self.prefix)
        return self.tmpdir

    def __exit__(self, type, value, traceback):
        if self.delete and self.tmpdir is not None:
            shutil.rmtree(self.tmpdir)


class RegisteryDecorator(object):
    """ Class that can be used to register class in a registry.
    """
    @classmethod
    def register(cls, obj_or_klass, *args, **kwargs):
        if "name" in kwargs:
            name = kwargs["name"]
        elif hasattr(obj_or_klass, "__name__"):
            name = obj_or_klass.__name__
        else:
            name = obj_or_klass.__class__.__name__
        if name in cls.REGISTRY:
            raise ValueError(
                "'{0}' name already used in registry.".format(name))
        cls.REGISTRY[name] = obj_or_klass
        return obj_or_klass

    @classmethod
    def get_registry(cls):
        return cls.REGISTRY


class Networks(RegisteryDecorator):
    """ Class that register all the available networks.
    """
    REGISTRY = {}


class Regularizers(RegisteryDecorator):
    """ Class that register all the available regularizers.
    """
    REGISTRY = {}


class Losses(RegisteryDecorator):
    """ Class that register all the available losses.
    """
    REGISTRY = {}


class Metrics(RegisteryDecorator):
    """ Class that register all the available losses.
    """
    REGISTRY = {}


def get_tools():
    """ List all available Deep Learning tools.

    Returns
    -------
    tools: dict
        all available tools for Deep Learning application.
    """
    tools = {}
    mod_members = dict(inspect.getmembers(sys.modules[__name__]))
    for key in ["Networks", "Regularizers", "Losses", "Metrics"]:
        tools[key.lower()] = mod_members[key].get_registry()
    return tools


def setup_logging(level="info", logfile=None):
    """ Setup the logging.

    Parameters
    ----------
    logfile: str, default None
        the log file.
    """
    logging_format = logging.Formatter(
        "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - "
        "%(message)s", "%Y-%m-%d %H:%M:%S")
    while len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[-1])
    while len(logger.handlers) > 0:
        logger.removeHandler(logger.handlers[-1])
    level = LEVELS.get(level, None)
    if level is None:
        raise ValueError("Unknown logging level.")
    logger.setLevel(level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(logging_format)
    logger.addHandler(stream_handler)
    if logfile is not None:
        file_handler = logging.FileHandler(logfile, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging_format)
        logger.addHandler(file_handler)
    if level != logging.DEBUG:
        warnings.simplefilter("ignore", DeprecationWarning)


def logo():
    """ pySAP logo is ascii art using Big Money-ne.

    Returns
    -------
    logo: str
        the logo.
    """
    logo = r"""
                                           /$$
                                          | $$
  /$$$$$$  /$$   /$$ /$$$$$$$   /$$$$$$  /$$$$$$
 /$$__  $$| $$  | $$| $$__  $$ /$$__  $$|_  $$_/
| $$  \ $$| $$  | $$| $$  \ $$| $$$$$$$$  | $$
| $$  | $$| $$  | $$| $$  | $$| $$_____/  | $$ /$$
| $$$$$$$/|  $$$$$$$| $$  | $$|  $$$$$$$  |  $$$$/
| $$____/  \____  $$|__/  |__/ \_______/   \___/
| $$       /$$  | $$
| $$      |  $$$$$$/
|__/       \______/ """
    return logo


def test_model(model, shape):
    """ Simple function to test a model.

    Parameters
    ----------
    model: Net
        the network model.
    shape: list of int
        the shape of a classical input batch dataset.
    """
    x = torch.autograd.Variable(torch.FloatTensor(np.random.random(shape)))
    out = model(x)
    loss = torch.sum(out)
    loss.backward()
    return out


def checkpoint(model, epoch, fold, outdir, optimizer=None, scheduler=None,
               **kwargs):
    """ Save the weights of a given model.

    Parameters
    ----------
    model: Net
        the network model.
    epoch: int
        the epoch index.
    fold: int
        the fold index.
    outdir: str
        the destination directory where a 'model_<fold>_epoch_<epoch>.pth'
        file will be generated.
    optimizer: Optimizer, default None
        the network optimizer (save the hyperparameters, etc.).
    scheduler: Scheduler, default None
        the network scheduler.
    kwargs: dict
        others parameters to save.
    """
    outfile = os.path.join(
        outdir, "model_{0}_epoch_{1}.pth".format(fold, int(epoch)))
    if optimizer is not None:
        kwargs.update(optimizer=optimizer.state_dict())
    if scheduler is not None:
        kwargs.update(scheduler=scheduler.state_dict())
    torch.save({
        "fold": fold,
        "epoch": epoch,
        "model": model.state_dict(),
        **kwargs}, outfile)
    return outfile


def get_named_layers(model, allowed_layers=ALLOWED_LAYERS, resume=False):
    """ Function that returned a dictionary with named layers.

    Parameters
    ----------
    model: Net
        the network model.
    allowed_layers: list of str, default ALLOWED_LAYERS
        the allowed modules.
    resume: bool, default False
        simplify layer names and skip type checking.

    Returns
    -------
    layers: dict
        the named layers.
    """
    layers = {}
    for name, mod in model.named_modules():
        name = name.replace("ops.", "")
        for klass in allowed_layers:
            if isinstance(mod, klass):
                if not resume:
                    if (hasattr(mod, "in_channels") and
                            hasattr(mod, "out_channels")):
                        name = "{0}-{1}.{2}".format(
                            name, mod.in_channels, mod.out_channels)
                    elif hasattr(mod, "num_features"):
                        name = "{0}-{1}".format(name, mod.num_features)
                    elif hasattr(mod, "in_features"):
                        name = "{0}-{1}".format(name, mod.in_features)
                    else:
                        raise ValueError("Layer of type '{0}' is not yet "
                                         "supported.".format(klass.__name__))
                layers[name] = mod
    return layers


def layer_at(model, layer_name, x, allowed_layers=ALLOWED_LAYERS):
    """ Access intermediate layers of pretrined network.

    Parameters
    ----------
    model: Net
        the network model.
    layer_name: str
        the layer name to be inspected.
    x: torch.Tensor
        an input tensor.
    allowed_layers: list of str, default ALLOWED_LAYERS
        the allowed modules.

    Returns
    -------
    hook_x: torch.Tensor
        the tensor generated at the requested location.
    weight: torch.Tensor
        the layer associated weight.
    """
    layers = get_named_layers(model)
    layer = layers[layer_name]
    global hook_x

    def hook(module, inp, out):
        """ Define hook.
        """
        print(
            "layer:", type(module),
            "\ninput:", type(inp),
            "\n   len:", len(inp),
            "\n   type:", type(inp[0]),
            "\n   data size:", inp[0].data.size(),
            "\n   data type:", inp[0].data.type(),
            "\noutput:", type(out),
            "\n   data size:", out.data.size(),
            "\n   data type:", out.data.type())
        global hook_x
        hook_x = out.data
    _hook = layer.register_forward_hook(hook)
    _ = model(x)
    _hook.remove()
    return hook_x.numpy(), layer.weight.detach().numpy()


def freeze_layers(model, layer_names):
    """ Freeze some wights in a network based on layer names.

    Parameters
    ----------
    model: Net
        the network model.
    layer_names: list of str
        the layer associated weights to be frozen.
    """
    layers = get_named_layers(model, allowed_layers=[torch.nn.Module],
                              resume=True)
    for name in layer_names:
        layer = layers[name]
        for param in layer.parameters():
            param.requires_grad = False


def reset_weights(model, checkpoint=None):
    """ Reset all the weights of a model. If a checkpoint is given, restore
    the checkpoint weights.

    Parameters
    ----------
    model: Net
        the network model.
    checkpoint: dict
        the saved model weights
    """
    def weight_reset(m):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    if checkpoint is None:
        model.apply(weight_reset)
    else:
        if hasattr(checkpoint, "state_dict"):
            model.load_state_dict(checkpoint.state_dict())
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)

def reset_optimizer(optimizer, checkpoint=None):
    """ Reset all the optimizer's parameters. If a checkpoint is given,
    restore the checkpoint parameters.

    Parameters
    ----------
    optimizer: Optimizer
        the network optimizer.
    checkpoint: dict
        the saved model weights
    """
    if isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        optimizer.load_state_dict(
            checkpoint["optimizer"])
    else:
        state_dict = optimizer.state_dict()
        state_dict['state'] = dict()
        state_dict['param_groups'] = [state_dict['param_groups'][0]]
        optimizer.load_state_dict(state_dict)
