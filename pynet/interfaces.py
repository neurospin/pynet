# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
This module provides a registery with ready to use networks that are
grouped into diferent categories: segmentation, registration, encoder, and
classifier.
"""


# Imports
import re
import logging
import textwrap
from pynet.core import Base


# Global parameters
logger = logging.getLogger("pynet")


def get_interfaces(family=None):
    """ List/sort all available Deep Learning training interfaces.

    Parameters
    ----------
    family: str or list of str, default None
        the interfaces family name.

    Returns
    -------
    interfaces: dict
        the requested interfaces.
    """
    if family is not None and not isinstance(family, list):
        family = [family]
    interfaces = {}
    for key in AVAILABLE_INTERFACES:
        klass = DeepLearningInterface.REGISTRY[key]
        if family is not None:
            for cnt, regex in enumerate(family):
                if re.match(regex, klass.__family__) is not None:
                    break
                cnt += 1
            if cnt == len(family):
                continue
        kname = klass.__name__
        interfaces.setdefault(klass.__family__, {})[kname] = klass
    return interfaces


class DeepLearningMetaRegister(type):
    """ Simple Python metaclass registry pattern.
    """
    REGISTRY = {}

    def __new__(cls, name, bases, attrs):
        """ Allocation.

        Parameters
        ----------
        name: str
            the name of the class.
        bases: tuple
            the base classes.
        attrs:
            the attributes defined for the class.
        """
        logger.debug("Adding '{0}' in registry...".format(name))
        new_cls = type.__new__(cls, name, bases, attrs)
        if name in cls.REGISTRY:
            raise ValueError(
                "'{0}' name already used in registry.".format(name))
        if name not in ("DeepLearningInterface", ):
            cls.REGISTRY[name] = new_cls
        logger.debug("  registry: {0}".format(cls.REGISTRY))
        return new_cls

    @classmethod
    def get_registry(cls):
        return cls.REGISTRY


class NetParameters(object):
    """ Put all the networks parameters to this class.
    You can do this during the init or by setting instance parameters or
    both.
    """
    def __init__(self, **kwargs):
        object.__setattr__(self, "net_kwargs", kwargs)

    def __setattr__(self, name, value):
        self.net_kwargs[name] = value

    def __repr__(self):
        return repr(self.net_kwargs)


class DeepLearningInterface(Base, metaclass=DeepLearningMetaRegister):
    """ Class to define ready to use Deep Learning interface for defined
    networks. An attributes section will be used for the documentation of the
    network parameters.
    """
    __family__ = None
    __net__ = None

    def __init__(self, net_params=None, pretrained=None, resume=False,
                 optimizer_name="Adam", learning_rate=1e-3,
                 loss_name="NLLLoss", metrics=None, use_cuda=False, **kwargs):
        """ Class initilization.

        Parameters
        ----------
        net_params: NetParameters, default None
            all the parameters that will be used during the network creation.
        pretrained: path, default None
            path to the pretrained model or weights.
        resume: bool, default False
            if set to true, the code will restore the weights of the model
            but also restore the optimizer's state, as well as the
            hyperparameters used, and the scheduler.
        optimizer_name: str, default 'Adam'
            the name of the optimizer: see 'torch.optim' for a description
            of available optimizer.
        learning_rate: float, default 1e-3
            the optimizer learning rate.
        loss_name: str, default 'NLLLoss'
            the name of the loss: see 'torch.nn' for a description
            of available loss.
        metrics: list of str
            a list of extra metrics that will be computed.
        use_cuda: bool, default False
            wether to use GPU or CPU.
        kwargs: dict
            specify directly a custom 'optimizer' or 'loss'. Can also be used
            to set specific optimizer parameters.
        """
        if self.__net__ is not None:
            logger.debug("Creating network '{0}'...".format(self.__net__))
            logger.debug("  family: {0}".format(self.__family__))
            logger.debug("  params: {0}".format(net_params))
            if net_params is None or not isinstance(net_params, NetParameters):
                raise ValueError("Please specify network parameters.")
            self.model = self.__net__(**net_params.net_kwargs)
        Base.__init__(
            self,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            loss_name=loss_name,
            metrics=metrics,
            use_cuda=use_cuda,
            pretrained=pretrained,
            resume=resume,
            **kwargs)


class DeepLearningDecorator(object):
    """ Decorator to determine the networks that need to be warped in the
    Deep Leanring interface environment.

    In order to make the class publicly accessible, we assign the result of
    the function to a variable dynamically using globals().
    """
    def __init__(self, family):
        """ Initialize the ValidationDecorator class.

        Parameters
        ----------
        family: str or list of str
            the families associated to the network.
        """
        self.destination_module_globals = globals()
        self.family = family
        if (not isinstance(self.family, list) and
                not isinstance(self.family, tuple)):
            self.family = [family]

    def __call__(self, klass, *args, **kwargs):
        """ Create the validator.

        Parameters
        ----------
        function: callable
            the function that perform the test.
        """
        for family in self.family:
            new_klass, klass_name = self._create_interface(klass, family)
            self.destination_module_globals[klass_name] = new_klass
        return klass

    def _create_interface(self, klass, family):
        """ Create the requested interface baed on the family name.
        """
        logger.debug("Creating interface for '{0}'...".format(klass))
        logger.debug("  family: {0}".format(family))
        category = family.title().replace(" ", "")
        logger.debug("  category: {0}".format(category))
        klass_name = klass.__name__ + category
        logger.debug("  class name: {0}".format(klass_name))
        mod_name = self.destination_module_globals["__name__"]
        logger.debug("  mod name: {0}".format(mod_name))
        doc = textwrap.dedent(klass.__doc__ or "")
        net_doc = klass.__init__.__doc__
        if net_doc is None:
            raise ValueError("Please specify the docstring of the model "
                             "__init__ method.")
        if "----------" not in net_doc:
            raise ValueError("Please specify the description of the network "
                             "parameters in the docstring of the model "
                             "__init__ method.")
        net_doc = textwrap.dedent(net_doc.split("----------")[1])
        net_doc = net_doc.strip("\n")
        doc += textwrap.dedent("""
        See the 'DeepLearningInterface' documentation for the  documentation
        of the generic parameters.
        The network parameters are discribed in the following section.

        Attributes
        ----------
        """)
        doc += net_doc
        class_parameters = {
            "__module__": mod_name,
            "_id":  mod_name + "." + klass_name,
            "__bases__": DeepLearningInterface.__bases__,
            "__mro__": DeepLearningInterface.__mro__,
            "__doc__": doc,
            "__family__": family,
            "__net__": klass
        }
        new_klass = type(
            klass_name, (DeepLearningInterface, ), class_parameters)

        return new_klass, klass_name


import pynet.models


AVAILABLE_INTERFACES = sorted(DeepLearningInterface.get_registry().keys())
