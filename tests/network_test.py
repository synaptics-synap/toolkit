# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

from pathlib import Path

from pysynap.network import parse_network, Tensor, TensorName, TensorShape, Operation, TensorId, OperationName, \
    create_dataflow_graph, TensorNode, OperationNode, OperationId

mobile_net_v2_path = ((Path(__file__)).parent.parent.parent.parent
                      / "models/image_classification/imagenet/model/"
                        "mobilenet_v2_1.0_224/mobilenet_v2_1.0_224_quant.tflite")


def test_parse(tmp_path):
    network = parse_network(mobile_net_v2_path, tmp_path)

    assert network

    assert network.tensors_manager.tensors["4"] == Tensor(name=TensorName('MobilenetV2/Conv_1/Relu6'),
                                                          shape=TensorShape([1, 7, 7, 1280]))

    assert network.operations_manager.operations["1"] == Operation(name=None,
                                                                   inputs=[TensorId('1')],
                                                                   outputs=[TensorId('11')])

    graph = create_dataflow_graph(network)

    assert len(graph.nodes) == 131
    assert len(graph.edges) == 140
    assert list(graph.edges)[0] == (TensorNode(TensorId("1")), OperationNode(OperationId("1")))
