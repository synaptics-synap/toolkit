# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

import json
import os
import zipfile
from pathlib import Path

from pysynap.bundler import Bundler
from pysynap.graph_segmenter import SubgraphTensor, SubgraphName, GlobalTensorId
from pysynap.meta import Delegate, DelegateInfo
from pysynap.network import TensorId, TensorShape, DataType


def create_mock_buffers(subgraph_idx: int):

    buf_meta_out_shape = TensorShape([(subgraph_idx + 1) * 10, (subgraph_idx + 2) * 10])

    buffer_in = SubgraphTensor(TensorId("0"), GlobalTensorId(str(subgraph_idx)), None,
                               TensorShape([subgraph_idx * 10, (subgraph_idx + 1) * 10]),
                               DataType.FLOAT32, None)

    buffer_out = SubgraphTensor(TensorId("1"), GlobalTensorId(str(subgraph_idx + 1)), None, buf_meta_out_shape,
                                DataType.FLOAT32, None)
    buffer_out2 = SubgraphTensor(TensorId("2"), GlobalTensorId(str(subgraph_idx + 10)), None, buf_meta_out_shape,
                                 DataType.FLOAT32, None)

    input_buffer = [buffer_in]
    output_buffer = [buffer_out, buffer_out2]

    return input_buffer, output_buffer


def create_stub_file(subgraph_dir: Path):

    meta_name = subgraph_dir / 'model.meta'

    with meta_name.open('w') as tf:
        json.dump({'test': 'Hello world'}, tf)

    model_name = subgraph_dir / 'model.nb'

    with model_name.open('wb') as tf:
        tf.write(b"test")

    return meta_name, model_name


def test_create(tmp_path):

    bundle_obj = Bundler()

    subgraph_ids = range(3)
    for i in subgraph_ids:
        subgraph_id = SubgraphName(f'subgraph_{i}')
        subgraph_dir = tmp_path / subgraph_id

        subgraph_dir.mkdir()

        meta_name, model_name = create_stub_file(subgraph_dir)

        input_buffers, output_buffers = create_mock_buffers(i)
        bundle_obj.add_subgraph(subgraph_name=subgraph_id,
                                input_tensors=input_buffers,
                                output_tensors=output_buffers,
                                model_filename=meta_name,
                                meta_filename=model_name,
                                delegate=DelegateInfo(Delegate.NPU, '') if i == 1 else DelegateInfo(Delegate.CPU, ''))

    bundle_path = tmp_path / 'test.synap'

    bundle_obj.inputs = ["0"]
    bundle_obj.outputs = ["12"]

    bundle_obj.create_bundle(bundle_path)

    assert bundle_path.exists()

    with zipfile.ZipFile(bundle_path) as z:
        assert set(z.namelist()) == {'bundle.json',
                                     '0/model.nb',
                                     '0/model.meta',
                                     '1/model.nb',
                                     '1/model.meta',
                                     '2/model.nb',
                                     '2/model.meta'}

        with z.open('bundle.json', 'r') as fp:
            data = json.load(fp)

        if 'UPDATE_TESTS' in os.environ:
            with (Path(__file__).parent / 'data' / 'bundle.json').open('w') as fp:
                json.dump(data, fp, indent=True)

        with (Path(__file__).parent / 'data' / 'bundle.json').open('r') as fp:
            expected_data = json.load(fp)

        assert data == expected_data
