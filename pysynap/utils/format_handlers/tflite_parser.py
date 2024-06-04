# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

from pathlib import Path
from copy import deepcopy
from typing import NewType

import pysynap.graph_segmenter as gs
from pysynap.external.tflite.tflite.Buffer import Buffer
from pysynap.external.tflite.tflite.Model import Model
from pysynap.external.tflite.tflite.Operator import Operator
from pysynap.external.tflite.tflite.TensorType import TensorType
from pysynap.external.tflite.tflite.SubGraph import SubGraph
from pysynap.graph_segmenter import SubgraphTensor, GlobalTensorId, QuantizationParameters
from pysynap.network import Network, Operation, TensorId, Tensor, TensorShape, OperationId, DataType
import logging

from pysynap.utils.format_handlers.tflite_writer import write_model


logger = logging.getLogger("synap.tflite")

"""
Represents a dictionary containing a JSON representation of a tflite file
"""
TfLiteSourceJson = NewType('TfLiteSourceJson', dict)


class TfliteNetwork(Network):
    """
    Network class that was parsed from tflite, used to keep track of the tflite internal representation
    """

    def __init__(self, source_model: Model) -> None:
        super().__init__()
        self.source_model = source_model


def load_tflite(model_path: Path, work_dir: Path) -> Model:
    """
    Loads the tflite file into memory using the python flatc generated classes
    """

    # TODO: here we could mmap the file to reduce the memory consumption and loading time
    with open(model_path, 'rb') as fp:
        return Model.GetRootAsModel(bytearray(fp.read()), 0)


def parse_tflite(model_path: Path, work_dir: Path) -> TfliteNetwork:
    """
    Parses a tflite file into a Network object
    """

    model = load_tflite(model_path, work_dir)

    if model.SubgraphsLength() != 1:
        raise Exception("Tflite file with more than one subgraphs not supported. "
                        f"Expected 1. Got {model.SubgraphsLength()}.")

    subgraph: SubGraph = model.Subgraphs(0)

    network = TfliteNetwork(model)

    tensors: set[int] = set()

    for idx in range(subgraph.OperatorsLength()):

        op: Operator = subgraph.Operators(idx)

        inputs = []

        for idx2 in range(op.InputsLength()):

            inp = op.Inputs(idx2)

            # inputs marked as -1 are optional and not used
            if inp == -1:
                continue

            buffer_id = subgraph.Tensors(inp).Buffer()

            buffer: Buffer = model.Buffers(buffer_id)

            # we need to skip the inputs that are constants
            if buffer.DataLength() > 0 or buffer.Size() > 0:
                logger.debug(f"Skipping input {subgraph.Tensors(inp).Name()} of operation {idx} "
                             "because it is a constant")
                continue

            inputs.append(TensorId(str(inp)))
            tensors.add(inp)

        outputs = [TensorId(str(op.Outputs(idx2))) for idx2 in range(op.OutputsLength())]
        tensors.update([op.Outputs(idx2) for idx2 in range(op.OutputsLength())])

        network.operations_manager.add_operation(OperationId(str(idx)), Operation(None, inputs, outputs))

    for x in range(subgraph.InputsLength()):
        input_id = subgraph.Inputs(x)
        network.meta_manager.inputs.append(TensorId(str(input_id)))
        tensors.add(input_id)

    for x in range(subgraph.OutputsLength()):
        output_id = subgraph.Outputs(x)
        network.meta_manager.outputs.append(TensorId(str(output_id)))
        tensors.add(output_id)

    for idx in range(subgraph.TensorsLength()):

        # skip tensors that are not input nor outputs of any operation
        if idx not in tensors:
            continue

        tensor = subgraph.Tensors(idx)
        shape = tensor.ShapeAsNumpy().tolist()

        network.tensors_manager.add_tensor(TensorId(str(idx)),
                                           Tensor(tensor.Name().decode('utf-8', 'backslashreplace'),  shape))

    return network


TFLITE_TO_SYNAP_TYPE: dict[str, DataType] = {
    TensorType.FLOAT32: DataType.FLOAT32,
    TensorType.FLOAT16: DataType.FLOAT16,
    TensorType.INT32: DataType.INT32,
    TensorType.UINT8: DataType.UINT8,
    TensorType.INT16: DataType.INT16,
    TensorType.INT8: DataType.INT8
}


def write_tflite_slices(segmented_network: gs.SegmentedNetwork,
                        output_dir: Path, work_dir: Path) -> list[gs.SerializedSubgraph]:
    """
    Writes out the slices of network as specified by the SegmentedNetwork object and returns
    a list of serialized subgraphs with the information about the graphs that were written out
    """

    if not isinstance(segmented_network.original_network, TfliteNetwork):
        raise Exception("Cannot slice non-tflite network")

    original_tflite_model = segmented_network.original_network.source_model

    output_dir.mkdir(parents=True, exist_ok=True)

    slices = []

    for subgraph_id, subgraph in segmented_network.subgraphs.items():
        logger.debug("slicing subgraph {}".format(subgraph_id))

        model_data, new_to_old_idx = write_model(original_tflite_model, subgraph.ops,
                                                 list(subgraph.inputs), list(subgraph.outputs))

        model_path = (output_dir / f"{subgraph_id}.tflite")

        with model_path.open('wb') as fp:
            fp.write(model_data)

        def create_subgraph_tensor(idx):

            tensor = original_tflite_model.Subgraphs(0).Tensors(int(idx))

            global_id = GlobalTensorId(idx)

            if tensor.Type() not in TFLITE_TO_SYNAP_TYPE:
                raise Exception(f"Type {tensor.Type()} of tensor {global_id} is not supported")

            qi = None

            if tensor.Quantization() is not None and tensor.Quantization().ScaleAsNumpy() != 0:
                if tensor.Quantization().ScaleLength() != 1:
                    raise Exception("Per axis quantization not supported")

                if tensor.Quantization().Details() is not None:
                    raise Exception(f"Quantization details of tensor {global_id} not supported")

                qi = QuantizationParameters(scale=tensor.Quantization().Scale(0),
                                            zero_point=tensor.Quantization().ZeroPoint(0))

            return SubgraphTensor(TensorId(str(idx)), global_id, tensor.Name().decode('utf-8', 'backslashreplace'),
                                  TensorShape(tensor.ShapeAsNumpy().tolist()),
                                  TFLITE_TO_SYNAP_TYPE[tensor.Type()], qi)

        inputs = [create_subgraph_tensor(x) for x in subgraph.inputs]
        outputs = [create_subgraph_tensor(x) for x in subgraph.outputs]

        slices.append(gs.SerializedSubgraph(
            name=subgraph_id, model_file=model_path,  weights_file=None, inputs=inputs, outputs=outputs))

    return slices
