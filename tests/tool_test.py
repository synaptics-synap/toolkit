# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

import json
import os
import subprocess
from pathlib import Path

from tests.conftest import tools_dir, vssdk_dir


def test_npu_convert(tmp_path, tflite_model_path):

    input_model_path = tflite_model_path
    input_meta_file = tmp_path / 'meta.yaml'

    with open(input_meta_file, 'w') as fp:
        json.dump({}, fp)

    ret = subprocess.run(["python3", "synap_convert.py",
                          "--model", str(input_model_path),
                          "--meta", str(input_meta_file),
                          "--target", "VS680",
                          "--out-dir", str(tmp_path),
                          "--out-format", "nb",
                          "--tools-dir", os.getenv("TOOLS_DIR", "path/to/dir"),
                          "--vssdk-dir", os.getenv("VSSDK_DIR", "path/to/dir")],
                         cwd=Path(__file__).parent.parent, capture_output=True, check=True)

    # check that the execution was silent
    assert ret.stdout == b''
    assert ret.stderr == b''

    output_metadata_path = tmp_path / "model.json"
    output_model_path = tmp_path / "model.nb"

    assert output_model_path.exists()
    assert output_metadata_path.exists()

    with output_metadata_path.open("r") as fp:
        output_metadata = json.load(fp)

    assert output_metadata == {'Inputs': {
        'input_0': {'name': 'serving_default_dense_input/0', 'shape': [1, 1], 'format': 'nhwc', 'dtype': 'float16',
                    'data_format': ''}}, 'Outputs': {
        'output_5': {'name': 'StatefulPartitionedCall/0', 'shape': [1, 1], 'format': 'nhwc', 'dtype': 'float16',
                     'data_format': ''}}, 'Recurrent_connections': {}, 'secure': False, 'delegate': 'npu'}

    assert output_model_path.stat().st_size == 32768


def test_npu_convert_verbose(tmp_path, tflite_model_path):

    input_model_path = tflite_model_path
    input_meta_file = tmp_path / 'meta.yaml'

    with open(input_meta_file, 'w') as fp:
        json.dump({}, fp)

    ret = subprocess.run(["python3", "synap_convert.py",
                          "--model", str(input_model_path),
                          "--meta", str(input_meta_file),
                          "--verbose",
                          "--target", "VS680",
                          "--out-dir", str(tmp_path),
                          "--out-format", "nb",
                          "--tools-dir", os.getenv("TOOLS_DIR", "path/to/dir"),
                          "--vssdk-dir", os.getenv("VSSDK_DIR", "path/to/dir")],
                         cwd=Path(__file__).parent.parent, capture_output=True, check=True)

    assert len(ret.stdout.splitlines()) > 0
    assert b'synap.acuitylib - DEBUG' not in ret.stdout  # something just in debug
    assert ret.stderr == b''


def test_npu_convert_debug(tmp_path, tflite_model_path):

    input_model_path = tflite_model_path
    input_meta_file = tmp_path / 'meta.yaml'

    with open(input_meta_file, 'w') as fp:
        json.dump({}, fp)

    ret = subprocess.run(["python3", "synap_convert.py",
                          "--model", str(input_model_path),
                          "--meta", str(input_meta_file),
                          "--debug",
                          "--target", "VS680",
                          "--out-dir", str(tmp_path),
                          "--out-format", "nb",
                          "--tools-dir", os.getenv("TOOLS_DIR", "path/to/dir"),
                          "--vssdk-dir", os.getenv("VSSDK_DIR", "path/to/dir")],
                         cwd=Path(__file__).parent.parent, capture_output=True, check=True)

    assert len(ret.stdout.splitlines()) > 0
    assert b'synap.acuitylib' in ret.stdout
    assert ret.stderr == b''


def test_npu_convert_fail(tmp_path, tflite_model_path):

    input_meta_file = tmp_path / 'meta.yaml'

    with open(input_meta_file, 'w') as fp:
        json.dump({}, fp)

    # call the tool with a corrupted model (in this case the metafile)
    ret = subprocess.run(["python3", "synap_convert.py",
                          "--model", str(input_meta_file),
                          "--meta", str(input_meta_file),
                          "--target", "VS680",
                          "--out-dir", str(tmp_path),
                          "--out-format", "nb",
                          "--tools-dir", os.getenv("TOOLS_DIR", "path/to/dir"),
                          "--vssdk-dir", os.getenv("VSSDK_DIR", "path/to/dir")],
                         cwd=Path(__file__).parent.parent, capture_output=True)

    # check that the execution failed
    assert ret.returncode != 0

    # check that the error messages are as expected
    assert ret.stdout == b''
    assert ret.stderr.startswith(b'Error: Unsupported model file: ') and len(ret.stderr.split(b'\n')) == 2

    # call the process again with a valid model, this time it shouldn't fail
    subprocess.check_call(["python3", "synap_convert.py",
                           "--model", str(tflite_model_path),
                           "--meta", str(input_meta_file),
                           "--target", "VS680",
                           "--out-dir", str(tmp_path),
                           "--out-format", "nb",
                           "--tools-dir", os.getenv("TOOLS_DIR", "path/to/dir"),
                           "--vssdk-dir", os.getenv("VSSDK_DIR", "path/to/dir")],
                          cwd=Path(__file__).parent.parent)


def test_hc_convert(tmp_path, tflite_model_path, tflite_hc_meta):
    input_model_path = tflite_model_path

    input_meta_file = tmp_path / 'meta.yaml'
    with open(input_meta_file, 'w') as fp:
        json.dump(tflite_hc_meta, fp)

    subprocess.check_call(["python3", "synap_convert.py",
                           "--model", str(input_model_path),
                           "--meta", str(input_meta_file),
                           "--target", "VS680",
                           "--out-dir", str(tmp_path),
                           "--out-format", "synap",
                           "--tools-dir", os.getenv("TOOLS_DIR", "path/to/dir"),
                           "--vssdk-dir", os.getenv("VSSDK_DIR", "path/to/dir")],
                          cwd=Path(__file__).parent.parent)

    output_model_path = tmp_path / "model.synap"

    assert output_model_path.exists()

    # Approximate size check
    assert 5500 < output_model_path.stat().st_size < 7000
