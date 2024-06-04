# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

import filecmp
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
import ruamel.yaml

from pysynap.meta import load_metafile
from tests.conftest import ReleaseModel, find_released_models, ConversionTask
import os


def test_find_release_models():
    """
    Checks that the find_find_released_models function finds all the models present in the release directory
    """

    found_released_models = [str(x.converted_model_path) for x in find_released_models()]
    found_released_models.sort()

    all_released_models = [str(x) for x in Path(os.environ["VSSDK_DIR"]).glob("synap/release/models/**/*.nb")]
    all_released_models.sort()

    assert found_released_models == all_released_models


def run_conversion_task(tmp_path, conversion_task: ConversionTask, meta_yaml: Path, out_format: str, monkeypatch):
    assert meta_yaml.exists()
    assert conversion_task.original_model_path.exists()

    extra_args = []

    if conversion_task.profiling:
        extra_args = ["--profiling"]

    if conversion_task.original_model_path.suffix == ".prototxt":
        extra_args = ["--weights", str(conversion_task.original_model_path /
                                       (conversion_task.original_model_path.suffix + ".caffemodel"))]

    with monkeypatch.context() as m:
        m.chdir(Path(__file__).parent.parent)
        m.setattr(sys, 'argv', ["synap_convert.py",
                           "--model", str(conversion_task.original_model_path),
                           "--meta", str(meta_yaml),
                           "--target", conversion_task.chip,
                           "--out-dir", str(tmp_path),
                           "--out-format", out_format,
                                "--tools-dir", str(tools_dir().absolute()),
                                "--vssdk-dir", str(vssdk_dir().absolute())] + extra_args)
        m.setenv("KEYS_BASE_DIR", str(conversion_task.key_base_dir.absolute()) + "/")
        m.setenv("CERTIFICATES_BASE_DIR", str(conversion_task.certificates_base_dir.absolute()) + "/")
        m.setenv("SYNAP_SECURITY_ENABLED", "true")

        monkeypatch.syspath_prepend(Path(__file__).parent)
        import synap_convert
        synap_convert.main()


def fixup_meta_yml(meta_yml: Path, new_meta_yml: Path, delegate: str):
    """
    Fixup the meta.yml file to add the delegate
    """
    with meta_yml.open('r') as fp:
        meta = ruamel.yaml.YAML().load(fp)

    if meta is None:
        meta = {}

    # ensure the models do not use security (we don't support it)
    if 'security' in meta:
        del meta['security']

    # tflite conversion doesn't support quantization at the moment
    if 'quantization' in meta:
        pytest.skip("Skipping test because the conversion task requires quantization")

    # set the delegate
    meta['delegate'] = {'*': delegate}

    # ensure all the format fields are expanded (no ${FILE} left)
    parsed_meta = load_metafile(meta_yml)
    for output, parsed_output in zip(meta.get('outputs', []), parsed_meta.outputs):
        if 'format' in output:
            output['format'] = parsed_output.data_format

    with new_meta_yml.open('w') as fp:
        ruamel.yaml.YAML().dump(meta, fp)


@pytest.mark.slow
def test_tflite_model(tmp_path, conversion_task: ConversionTask, monkeypatch):

    # do not try to convert models that are not tflite
    if conversion_task.original_model_path.suffix not in ['.tflite']:
        pytest.skip("Skipping test because the input model is not supported")

    if conversion_task.chip != 'VS680A0':
        pytest.skip("Skipping test because it is the same as on VS680A0")

    fixup_meta_yml(conversion_task.meta_yaml_path, tmp_path / "meta.yaml", 'gpu')

    run_conversion_task(tmp_path, conversion_task, tmp_path / "meta.yaml", 'synap', monkeypatch)

    assert (tmp_path / f"{conversion_task.original_model_path.stem}.synap").exists()


@pytest.mark.slow
def test_npu_model(tmp_path, conversion_task: ConversionTask, monkeypatch):

    fixup_meta_yml(conversion_task.meta_yaml_path, tmp_path / "meta.yaml", 'npu')

    run_conversion_task(tmp_path, conversion_task, tmp_path / "meta.yaml", 'synap', monkeypatch)

    assert (tmp_path / f"{conversion_task.original_model_path.stem}.synap").exists()


@pytest.mark.slow
def test_release_model(tmp_path, release_model: ReleaseModel, monkeypatch):
    assert release_model.converted_model_path.exists()

    # FIXME: we have a small difference in the scale for some models and the model itself
    if release_model.conversion_task.original_model_path.name in ['blur_quant.tflite',
                                                                  'yolov5s_face_640x480_onnx.onnx']:
        pytest.skip("Model not supported yet")

    run_conversion_task(tmp_path, release_model.conversion_task,
                        release_model.conversion_task.meta_yaml_path, 'nb', monkeypatch)

    assert (tmp_path / "model.nb").exists()
    assert (tmp_path / "model.json").exists()

    with open(tmp_path / "model.json") as fp:
        new_metadata = json.load(fp)

    with open(release_model.converted_model_meta_path) as fp:
        old_metadata = json.load(fp)

    # released models do not have delegate yet
    del new_metadata['delegate']

    assert new_metadata == old_metadata

    # when the model is secure each run has a different encryption seed, so they are all different, we can only
    # check the size
    if new_metadata['secure']:
        assert (tmp_path / "model.nb").stat().st_size == release_model.converted_model_path.stat().st_size
    else:
        assert filecmp.cmp((tmp_path / "model.nb"), release_model.converted_model_path)
