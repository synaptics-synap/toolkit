# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

from pathlib import Path

from pysynap.meta import Delegate, DelegateInfo, load_metafile


def test_load(tmp_path):
    meta = load_metafile(Path(__file__).parent / 'data' / 'meta.yaml')

    assert meta.delegates == {
        "*": DelegateInfo(Delegate.CPU, ''),
        "4": DelegateInfo(Delegate.NPU, ''),
        "5": DelegateInfo(Delegate.NPU, ''),
        "10": DelegateInfo(Delegate.GPU, '')
    }

    assert meta.inputs[0].name == "MobilenetV2/expanded_conv_1/expand/Relu6"
    assert meta.inputs[0].shape == [1, 112, 112, 96]
    assert meta.outputs[0].name == "MobilenetV2/expanded_conv_3/expand/Relu6"
