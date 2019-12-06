# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import os
import subprocess

currentdir = os.path.dirname(__file__)
examplesdir = os.path.join(currentdir, os.pardir, "examples")

example_files = []
for root, dirs, files in os.walk(examplesdir):
    for basneame in files:
        if basneame.endswith(".py"):
             example_files.append(os.path.abspath(
                os.path.join(root, basneame)))
print("'{0}' examples found!".format(len(example_files)))

for path in example_files:
    print("-- ", path)
    cmd = ["python3", path]
    subprocess.check_call(cmd, env=os.environ)
