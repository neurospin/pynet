.. -*- mode: rst -*-

|PythonVersion|_ |Coveralls|_ |Travis|_ |PyPi|_ |Doc|_ |CircleCI|_

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue
.. _PythonVersion: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue

.. |Coveralls| image:: https://coveralls.io/repos/neurospin/pynet/badge.svg?branch=master&service=github
.. _Coveralls: https://coveralls.io/github/neurospin/pynet

.. |Travis| image:: https://travis-ci.com/neurospin/pynet.svg?branch=master
.. _Travis: https://travis-ci.com/neurospin/pynet

.. |PyPi| image:: https://badge.fury.io/py/python-network.svg
.. _PyPi: https://badge.fury.io/py/python-network

.. |Doc| image:: https://readthedocs.org/projects/python-network/badge/?version=latest
.. _Doc: https://python-network.readthedocs.io/en/latest/?badge=latest

.. |CircleCI| image:: https://circleci.com/gh/neurospin/pynet.svg?style=svg
.. _CircleCI: https://circleci.com/gh/neurospin/pynet



.. image:: https://github.com/neurospin/pynet/blob/master/doc/source/_static/pynet.png
    :width: 400px


Helper Module for Deep Learning with pytorch.

This work is made available by a `community of people
<https://github.com/neurospin/pynet/blob/master/AUTHORS.rst>`_, amoung which the
CEA Neurospin BAOBAB laboratory.

Important links
===============

- `Official source code repo <https://github.com/neurospin/pynet>`_
- `HTML documentation <http://neurospin.github.io/pynet>`_
- `Release notes <https://github.com/neurospin/pynet/blob/master/CHANGELOG.rst>`_

Where to start
==============

You can list all available Deep Learning tools by executing in a Python shell::

    from pprint import pprint
    import pynet
    pprint(pynet.get_tools())

The 'get_tools' function returns a dictionary with all available 'networks',
'losses', 'regularizers', and 'metrics'.

Then each network has been embeded in a Deep Learning training interface
providing a 'training' and a 'testing' method.
Network parameters are set using the NetParameters object.
You can list all these interfaces by executing in a Python shell::

    from pprint import pprint
    import pynet
    pprint(pynet.get_interfaces(family=None))
    params = pynet.NetParameters(param1=1, param2=2)
    params.param3 = 3

The 'get_interfaces' function returns a dictionary with interfaces sorted by
family names. You can filter the result by providing the family name or a list
of family names of interest.

You can list also all available data fetchers by executing in a Python shell::

    from pprint import pprint
    import pynet.datasets import get_fetchers
    pprint(get_fetchers())

The 'get_fetchers' function returns a dictionary with all the declared
fetchers. Finally you may want to look at the data manger class that provides
convenient tools to split/stratify your dataset::

    from pynet.datasets import DataManager

Install
=======

Make sure you have installed all the package dependencies.
Further instructions are available `here
<https://python-network.readthedocs.io/en/latest/generated/installation.html>`_.






