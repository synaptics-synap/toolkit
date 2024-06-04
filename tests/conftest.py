# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
import subprocess
import os
import logging
from _pytest.compat import NotSetType

logging.getLogger("synap").setLevel(logging.DEBUG)


class Board:

    def __init__(self, request):
        self.request = request

    @contextmanager
    def start(self, command):
        p = subprocess.Popen(["adb", "shell"] + command)

        try:
            yield p
        finally:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()

    def push(self, src, dst, cache=False):
        if cache and self.check_file(dst):
            return
        subprocess.check_call(["adb", "push", src, dst])

    def check_file(self, path):
        return (self.get_shell_output(["test", "-f", path, "&&", "echo", "found", "||", "echo", "not found"])
                == b'found\n')

    @staticmethod
    def pull(src, dst):
        subprocess.check_call(["adb", "pull", src, dst])

    @staticmethod
    def get_shell_output(command):
        return subprocess.check_output(["adb", "shell"] + command)

    @staticmethod
    def shell(command):
        return subprocess.check_call(["adb", "shell"] + command)

    def is_root(self):
        return self.get_shell_output(['id', '-u']) == b'0\n'


@pytest.fixture(scope='session')
def board(request):
    b = Board(request)

    assert b.is_root()

    b.shell(['setenforce', '0'])

    return b


@pytest.fixture(scope='session')
def tflite_model():
    # delay the import so that we can silence the deprecation warnings, and we speed up the loading of tests when
    # tf is not needed
    import tensorflow as tf

    # create a model using high-level tf.keras.* APIs
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=[1]),
        tf.keras.layers.Dense(units=16, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])

    # compile the model
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(x=[-1, 0, 1], y=[-3, -1, 1], epochs=5)

    # convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()


@pytest.fixture
def tflite_model_path(tflite_model, tmp_path):
    model_path = tmp_path / 'model.tflite'
    with open(model_path, 'wb') as fp:
        fp.write(tflite_model)

    return model_path


@pytest.fixture
def tflite_hc_meta():
    return {"delegate": {"*": "cpu", "1": "gpu"}}


def tools_dir():
    return Path(os.environ['TOOLS_DIR'])


def vssdk_dir():
    return Path(os.environ['VSSDK_DIR'])


def models_dir():
    return Path(os.environ['MODELS_DIR'])


@dataclass
class ConversionTask:
    """
    Represents a model and associated conversion meta.yml that is available in the SyNAP model zoo
    """
    chip: str
    original_model_path: Path
    meta_yaml_path: Path
    key_base_dir: Path
    certificates_base_dir: Path
    profiling: bool

    @property
    def name(self):
        name = self.meta_yaml_path.stem.replace('@', '_')
        if name == 'model_metafile':
            name = self.original_model_path.stem
        return name


def find_conversion_tasks():
    """
    Find all meta.yml files in the model zoo and information abut the conversion task they specify
    """

    vssdk_root = vssdk_dir()
    models_root = models_dir()

    models = []

    chips = [chip_dir.name for chip_dir in (vssdk_root / 'synap' / 'release' / 'models').iterdir()]

    chip_dirs = {
        "VS640A0": "vs640/A0/",
        "VS680A0": "vs680/A0/"
    }

    key_base_dirs = {chip: vssdk_root / "security/keys/chip" / chip_dirs[chip] for chip in chips}
    certificates_base_dirs = {chip: vssdk_root / "configs/product/common" / chip_dirs[chip] for chip in chips}

    models_files = [x for x in models_root.glob("**/*.tflite")]
    models_files += [x for x in models_root.glob("**/*.onnx")]
    models_files += [x for x in models_root.glob("**/*.pb")]
    models_files += [x for x in models_root.glob("**/*.prototxt")]
    models_files = [x for x in models_files if not x.name.endswith("_float.tflite")]
    models_files = [x for x in models_files if not x.stem.endswith("_disabled")]

    for model_file in models_files:

        for chip in chips:

            if (model_file.parent / f"{model_file.stem}.yaml").exists():
                # Use companion .yaml file
                metafiles = [model_file.parent / f"{model_file.stem}.yaml"]
            elif (model_file.parent / model_file.stem).is_dir():
                # Use all .yaml files in companion directory
                metafiles = list((model_file.parent / model_file.stem).glob("*.yaml"))
            else:

                metafiles = []

                # Check for default metafile in current and parent directories
                for prefix in [".", "..", "../..", "../../.."]:
                    if (model_file.parent / prefix / "model_metafile.yaml").exists():
                        metafiles = [model_file.parent / prefix / "model_metafile.yaml"]
                        break

            for metafile in metafiles:
                models.append(ConversionTask(chip=chip, original_model_path=model_file,
                                             meta_yaml_path=metafile,
                                             profiling=metafile.name.endswith("_profiling.yaml"),
                                             key_base_dir=key_base_dirs[chip],
                                             certificates_base_dir=certificates_base_dirs[chip]))

    return models


@dataclass
class ReleaseModel:
    """
    Represents a model that is released in the vssdk/synap/release directory
    """
    converted_model_path: Path
    converted_model_meta_path: Path
    conversion_task: ConversionTask


def find_released_models():
    """
    Find the models that are present in the release directory and the corresponding conversion task
    """

    vssdk_root = vssdk_dir()
    models_root = models_dir()

    models = []

    for conversion_task in find_conversion_tasks():

        model_file = conversion_task.original_model_path

        task_category = model_file.relative_to(models_root).parts[0]
        task_name = model_file.relative_to(models_root).parts[1]

        chip = conversion_task.chip

        metafile = conversion_task.meta_yaml_path

        converted_model_name = metafile.stem

        if converted_model_name == 'model_metafile':
            converted_model_name = model_file.stem

        compiled_model = (vssdk_root / 'synap' / 'release' / 'models' / chip /
                          task_category / task_name / "model" / converted_model_name / "model.nb")

        if not compiled_model.is_file():
            continue

        compiled_model_meta = compiled_model.parent / (compiled_model.stem + ".json")

        if not compiled_model_meta.is_file():
            continue

        models.append(ReleaseModel(conversion_task=conversion_task,
                                   converted_model_path=compiled_model,
                                   converted_model_meta_path=compiled_model_meta))

    return models


def generate_conversion_task_id(m: ConversionTask):
    # this is required in case the parameter is not set (due to the fact no conversion tasks have been found)
    if isinstance(m, NotSetType):
        return m
    return m.chip + "_" + m.name + "_" + m.original_model_path.suffix[1:]


def generate_release_model_task_id(m: ReleaseModel):
    # this is required in case the parameter is not set (due to the fact no release models have been found)
    if isinstance(m, NotSetType):
        return m
    return generate_conversion_task_id(m.conversion_task)


def pytest_generate_tests(metafunc):
    # here we generate a list of all released models that can be used as test parameter to detect regressions
    if "release_model" in metafunc.fixturenames:
        metafunc.parametrize("release_model", find_released_models(), ids=generate_release_model_task_id)

    # here we generate a list of all conversion tasks in the model zoo so that we can detect unsupported models
    if "conversion_task" in metafunc.fixturenames:
        metafunc.parametrize("conversion_task", find_conversion_tasks(), ids=generate_conversion_task_id)


def pytest_addoption(parser):
    # define a cli option to enable execution of slow tests
    parser.addoption("--runslow", action="store_true", help="run slow tests")


def pytest_configure(config):
    # define a marker that we use to mark slow tests, we can use this to skip them
    config.addinivalue_line("markers", "slow: mark test as slow to run")

    # ignore deprecation warnings from distutils (raised by tf)
    config.addinivalue_line("filterwarnings", "ignore:.*distutils package is deprecated.*:DeprecationWarning")


def pytest_collection_modifyitems(config, items):
    """
    This function is executed after the list of test to run has been created by pytest
    """

    # when --runslow is not given in cli skip slow tests
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
