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
        klass = DeepLearningFramework.REGISTRY[key]
        if family is not None:
            for cnt, regex in enumerate(family):
                if re.match(regex, klass.__family__) is not None:
                    break
                cnt += 1
            if cnt == len(family):
                continue
        interfaces.setdefault(klass.__family__, []).append(klass)
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
        if name not in ("DeepLearningFramework", ):
            cls.REGISTRY[name] = new_cls
        logger.debug("  registry: {0}".format(cls.REGISTRY))
        return new_cls

    @classmethod
    def get_registry(cls):
        return cls.REGISTRY


class DeepLearningFramework(Base, metaclass=DeepLearningMetaRegister):
    """ Class to define ready to use Deep Learning framework for defined
    networks. An attributes section will be used for the documentation of the
    network parameters.
    """
    __family__ = None
    __net__ = None

    def __init__(self, net_kwargs,
                 pretrained=None, optimizer_name="Adam",
                 learning_rate=1e-3, loss_name="NLLLoss", metrics=None,
                 use_cuda=False, **kwargs):
        """ Class initilization.

        Parameters
        ----------
        net_kwargs: dict
            all the parameters associated with the network creation.
        pretrained: path, default None
            path to the pretrained model or weights.
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
        logger.debug("Creating network '{0}'...".format(self.__net__))
        logger.debug("  family: {0}".format(self.__family__))
        logger.debug("  kwargs: {0}".format(net_kwargs))
        self.model = self.__net__(**net_kwargs)
        Base.__init__(
            self,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            loss_name=loss_name,
            metrics=metrics,
            use_cuda=use_cuda,
            pretrained=pretrained,
            **kwargs)


class DeepLearningDecorator(object):
    """ Decorator to determine the netorks that need to be warped in a
    Deep Leanring framework environment.

    In order to make the class publicly accessible, we assign the result of
    the function to a variable dynamically using globals().
    """
    def __init__(self, family):
        """ Initialize the ValidationDecorator class.

        Parameters
        ----------
        family: str
            the family associated to the network.
        """
        self.destination_module_globals = globals()
        self.family = family

    def __call__(self, klass, *args, **kwargs):
        """ Create the validator.

        Parameters
        ----------
        function: callable
            the function that perform the test.
        """
        logger.debug("Creating framework for '{0}'...".format(klass))
        logger.debug("  family: {0}".format(self.family))
        category = self.family.title().replace(" ", "")
        logger.debug("  category: {0}".format(category))
        class_name = klass.__name__ + category
        logger.debug("  class name: {0}".format(class_name))
        mod_name = self.destination_module_globals["__name__"]
        logger.debug("  mod name: {0}".format(mod_name))
        doc = textwrap.dedent(klass.__doc__)
        net_doc = textwrap.dedent(
            klass.__init__.__doc__.split("----------")[1])
        net_doc = net_doc.strip("\n")
        doc += textwrap.dedent("""
        The network kwargs are discribed in the following section.

        Attributes
        ----------
        """)
        doc += net_doc
        class_parameters = {
            "__module__": mod_name,
            "_id":  mod_name + "." + class_name,
            "__bases__": DeepLearningFramework.__bases__,
            "__mro__": DeepLearningFramework.__mro__,
            "__doc__": doc,
            "__family__": self.family,
            "__net__": klass
        }
        new_klass = type(
            class_name, (DeepLearningFramework, ), class_parameters)
        self.destination_module_globals[class_name] = new_klass
        return klass


import pynet.models


AVAILABLE_INTERFACES = sorted(DeepLearningFramework.get_registry().keys())
