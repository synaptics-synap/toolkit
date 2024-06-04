# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

import filecmp
import subprocess
from pathlib import Path

from pysynap.utils.format_handlers.tflite_parser import load_tflite
from pysynap.utils.format_handlers.tflite_writer import write_model

mobile_net_v2_path = ((Path(__file__)).parent.parent.parent.parent
                      / "models/image_classification/imagenet/model/"
                        "mobilenet_v2_1.0_224/mobilenet_v2_1.0_224_quant.tflite")


def test_write(tmp_path):
    tflite = load_tflite(mobile_net_v2_path, tmp_path)

    data, _ = write_model(tflite)

    with mobile_net_v2_path.open('rb') as fp:
        expected_data = bytearray(fp.read())

    out_file = tmp_path / 'test.tflite'

    with (tmp_path / 'test.tflite').open('wb') as fp:
        fp.write(data)

    schema = Path(__file__).parent.parent / 'pysynap' / 'external' / 'tflite' / 'schema.fbs'

    subprocess.check_call(['flatc', '-t', '--strict-json', '--defaults-json',
                           '-o', str(tmp_path / 'output_gen'), str(schema), '--', str(out_file)])

    subprocess.check_call(['flatc', '-t', '--strict-json', '--defaults-json',
                           '-o', str(tmp_path / 'output_orig'), str(schema), '--', str(mobile_net_v2_path)])

    assert filecmp.cmp(tmp_path / 'output_gen' / (out_file.stem + '.json'),
                       tmp_path / 'output_orig' / (mobile_net_v2_path.stem + '.json'))

    #assert len(data) == len(expected_data)

    #assert data == expected_data
