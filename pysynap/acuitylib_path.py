# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

import os
import sys
import pathlib

toolkit_dir = pathlib.Path(__file__).parent.parent.absolute()
internal_tree = False
# check if pysynap is run from the source tree
vsi_acuity_dir = f"{toolkit_dir}/../../acuity"
if os.path.exists(f"{vsi_acuity_dir}/acuitylib/__init__.py"):
    internal_tree = True
else:
    # Check if toolkit-prebuilts tree is available alongside
    vsi_acuity_dir  = f"{toolkit_dir}/../toolkit-prebuilts"
    acuitylib_init = f"{vsi_acuity_dir}/acuitylib/__init__.py"
    if not os.path.exists(acuitylib_init):
        raise Exception("Cannot find %s " % acuitylib_init)

sys.path.append(os.path.abspath(vsi_acuity_dir))
