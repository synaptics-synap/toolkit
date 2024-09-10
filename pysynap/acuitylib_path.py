# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

import os
import sys
import pathlib

current_path = pathlib.Path(__file__)
toolkit_dir = current_path.parent.parent.absolute()
internal_tree = False
# check if pysynap is run from the source tree
vsi_acuity_dir = f"{toolkit_dir}/../../acuity"
if os.path.exists(f"{vsi_acuity_dir}/acuitylib/__init__.py"):
    internal_tree = True
#check if prebuilts tree is available in current directory
elif os.path.exists(f"{current_path.parent.absolute()}/prebuilts/acuitylib/__init__.py"):
    vsi_acuity_dir  = f"{current_path.parent.absolute()}/prebuilts/"
else:
    # Check if toolkit-prebuilts tree is available alongside
    vsi_acuity_dir  = f"{toolkit_dir}/../toolkit-prebuilts"
    acuitylib_init = f"{vsi_acuity_dir}/acuitylib/__init__.py"
    if not os.path.exists(acuitylib_init):
        raise Exception("Cannot find %s " % acuitylib_init)

sys.path.append(os.path.abspath(vsi_acuity_dir))