# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

import json
import os
from pathlib import Path
from zipfile import ZipFile

import tensorflow as tf

from pysynap.converters import ConversionOptions, TargetSOC
from pysynap.meta import load_metafile
from pysynap.converter import HeterogeneousConverter
import numpy as np

from pysynap.network import DataType


def infer_with_bundle(bundle_path, input_data):
    with ZipFile(bundle_path) as zf:

        with zf.open('bundle.json', 'r') as fp:
            bundle_info = json.load(fp)

        assert len(bundle_info["graph"]) == 3

        for idx, subgraph in enumerate(bundle_info['graph']):
            print(f"inferring with {idx} input {input_data}")
            assert subgraph['inputs'] == [{'subgraph': idx - 1, 'out': 0}]

            with zf.open(subgraph['meta'], 'r') as fp:
                model_meta = json.loads(fp.read().decode())

            for input_meta in model_meta['Inputs'].values():
                assert input_meta['dtype'] == DataType.FLOAT32.value

            for input_meta in model_meta['Outputs'].values():
                assert input_meta['dtype'] == DataType.FLOAT32.value

            with zf.open(subgraph['model'], 'r') as fp:
                tflite_model = fp.read()

                input_data = infer_with_tflite(tflite_model, input_data)

        return input_data


def infer_with_tflite(tflite_model, input_data):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    return interpreter.get_tensor(output_details[0]['index'])


def test_e2e(tmp_path, tflite_model, tflite_hc_meta):

    # save the model
    input_model_path = tmp_path / 'model.tflite'
    with open(input_model_path, 'wb') as f:
        f.write(tflite_model)

    input_meta_file = tmp_path / 'meta.yaml'
    with open(input_meta_file, 'w') as fp:
        json.dump(tflite_hc_meta, fp)

    meta = load_metafile(input_meta_file)

    conversion_options = ConversionOptions(target=TargetSOC.VS680, vssdk_dir=Path('path/to/vssdk'),
                                           tools_dir=Path(os.getenv('TOOLS_DIR', 'path/to/tools')),
                                           profiling=False, cpu_profiling=False,
                                           verbose=True, debug=False)

    converter = HeterogeneousConverter(conversion_options)

    output_model = tmp_path / 'model.synap'

    converter.convert(input_model_path, None, meta, output_model, tmp_path)

    # compare inference output of split and original models
    input_data = np.ones(shape=(1, 1), dtype=np.float32)

    expected_output = infer_with_tflite(tflite_model, input_data)
    output = infer_with_bundle(output_model, input_data)

    assert expected_output == output
