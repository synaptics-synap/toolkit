# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

import dataclasses
import filecmp
import json
from pathlib import Path
import os
import shutil

from pysynap.meta import load_metafile, MetaInfo, Delegate, DelegateInfo
from pysynap.graph_segmenter import segment, SerializedSegmentedNetwork, SubgraphName
from pysynap.network import WILDCARD_OPERATION_PATTERN, OperationPattern
from pysynap.utils.format_handlers.onnx_parser import load_onnx

from pysynap.utils.format_handlers.tflite_parser import load_tflite

mobile_net_v2_path = ((Path(__file__)).parent.parent.parent.parent
                      / "models/image_classification/imagenet/model/"
                        "mobilenet_v2_1.0_224/mobilenet_v2_1.0_224_quant.tflite")


def validate_results(subgraph_details: SerializedSegmentedNetwork, reference_file: Path):

    def to_json_data(x):
        if dataclasses.is_dataclass(x):
            field_names = [z.name for z in dataclasses.fields(x)]
            return {y: to_json_data(getattr(x, y)) for y in field_names}
        elif isinstance(x, dict):
            return {y: to_json_data(x[y]) for y in x}
        elif isinstance(x, list):
            return [to_json_data(y) for y in x]

        return str(x)

    details_dict = to_json_data(subgraph_details)

    # delete the model and weight file names as they will keep changing
    for subgraph in details_dict['subgraphs'].values():
        del subgraph['model_file']
        del subgraph['weights_file']

    if 'UPDATE_TESTS' in os.environ:
        with reference_file.open('w') as fp:
            json.dump(details_dict, fp, indent=True)

    with reference_file.open('r') as fp:
        expected_data = json.load(fp)

    assert expected_data == details_dict


def test_split_tflite(tmp_path):
    meta = load_metafile(Path(__file__).parent / 'data' / 'meta.yaml')

    subgraph_details = segment(network_path=mobile_net_v2_path, meta=meta, default_delegate=DelegateInfo(Delegate.NPU, ''), output_path=tmp_path, work_dir=tmp_path)

    validate_results(subgraph_details, (Path(__file__).parent / 'data' / 'subgraph_details_tflite.json'))

    for i in range(3):

        create_path = subgraph_details.subgraphs[SubgraphName(f'subgraph_{i}')].model_file
        expected_path = Path(__file__).parent / 'data' / 'split_tflite' / f'subgraph_{i}' / f'subgraph_{i}.tflite'

        if 'UPDATE_TESTS' in os.environ:
            os.makedirs(expected_path.parent, exist_ok=True)
            shutil.copyfile(create_path, expected_path)

        assert filecmp.cmp(expected_path, create_path), f"model {i} tflite is not correct"


age_googlenet_path = ((Path(__file__)).parent.parent.parent.parent
                      / "models/image_classification/age/model/"
                        "age_googlenet_onnx/age_googlenet.onnx")


def test_split_onnx(tmp_path):
    meta = MetaInfo()

    meta.delegates[WILDCARD_OPERATION_PATTERN] = DelegateInfo(Delegate.CPU, '')
    meta.delegates[OperationPattern("9")] = DelegateInfo(Delegate.GPU, '')

    subgraph_details = segment(network_path=age_googlenet_path, meta=meta, default_delegate=DelegateInfo(Delegate.NPU, ''), output_path=tmp_path, work_dir=tmp_path)

    validate_results(subgraph_details, (Path(__file__).parent / 'data' / 'subgraph_details_onnx.json'))

    for i in range(3):

        create_path = subgraph_details.subgraphs[SubgraphName(f'subgraph_{i}')].model_file
        expected_path = Path(__file__).parent / 'data' / 'split_onnx' / f'subgraph_{i}' / f'subgraph_{i}.onnx'

        if 'UPDATE_TESTS' in os.environ:
            os.makedirs(expected_path.parent, exist_ok=True)
            shutil.copyfile(create_path, expected_path)

        created = load_onnx(create_path)
        expected = load_onnx(expected_path)

        assert created == expected, f"model {i} onnx is not correct"
