# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

import json
import os
from pathlib import Path
from zipfile import ZipFile

from pysynap.converter import HeterogeneousConverter
from pysynap.converters import ConversionOptions, TargetSOC
from pysynap.meta import load_metafile, Delegate, DelegateInfo

mobile_net_v2_path = ((Path(__file__)).parent.parent.parent.parent
                      / "models/image_classification/imagenet/model/"
                        "mobilenet_v2_1.0_224/mobilenet_v2_1.0_224_quant.tflite")


def test_model_json_input_scale(tmp_path):
    meta = load_metafile(Path(__file__).parent / 'data' / 'meta.yaml')

    meta.inputs[0].means = 0
    meta.inputs[0].scale = 1

    meta.delegates = {'*': DelegateInfo(Delegate.GPU, '')}

    options = ConversionOptions(target=TargetSOC.VS680, vssdk_dir=Path('path/to/vssdk'),
                                tools_dir=Path(os.getenv('TOOLS_DIR', 'path/to/tools')),
                                profiling=False, cpu_profiling=False,
                                verbose=True, debug=False)

    converter = HeterogeneousConverter(conversion_options=options)

    model_path = tmp_path / 'model.synap'

    converter.convert(mobile_net_v2_path, None, meta, model_path, tmp_path)

    assert model_path.exists()

    with ZipFile(model_path) as zf:
        with zf.open('0/subgraph_0.json') as fp:
            data = json.load(fp)

    assert list(data['Inputs'].values())[0]['mean'] == 0
    assert list(data['Inputs'].values())[0]['scale'] == 1
    assert list(data['Inputs'].values())[0]['format'] == 'nhwc'
    assert list(data['Inputs'].values())[0]['name'] == meta.inputs[0].name
    assert list(data['Inputs'].values())[0]['data_format'] == meta.inputs[0].data_format or meta.input_format


def test_convert(tmp_path):
    meta = load_metafile(Path(__file__).parent / 'data' / 'meta.yaml')

    options = ConversionOptions(target=TargetSOC.VS680, vssdk_dir=Path('path/to/vssdk'),
                                tools_dir=Path(os.getenv('TOOLS_DIR', 'path/to/tools')),
                                profiling=False, cpu_profiling=False,
                                verbose=True, debug=False)

    converter = HeterogeneousConverter(conversion_options=options)

    model_path = tmp_path / 'model.synap'

    converter.convert(mobile_net_v2_path, None, meta, model_path, tmp_path)

    assert model_path.exists()
