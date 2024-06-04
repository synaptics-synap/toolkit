# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

import itertools
import tempfile
from pathlib import Path

from onnx import (
    load,
    save,
    utils,
    shape_inference,
    ModelProto,
    onnx_cpp2py_export,
    checker
)

from onnxruntime.tools import (
    onnx_model_utils
)

import pysynap.graph_segmenter as gs
from pysynap.graph_segmenter import SubgraphTensor, GlobalTensorId
from pysynap.exceptions import ConversionError
from pysynap.network import (
    Network,
    Operation,
    TensorId,
    Tensor,
    TensorShape,
    OperationId,
    DataType
)

import logging
logger = logging.getLogger("synap.onnx")

onnx_dtype_list = ["UNDEFINED", "float32", "uint8", "int8", "uint16", "int16", "int32", "int64",
              "string", "bool", "float16", "float64", "uint32", "uint64", "complex64", "complex128",
             "bfloat16", "f8E4M3FN", "f8E4M3FNUZ", "f8E5M2", "f8E5M2FNUZ"]

# TODO: SYNAP_TYPE is less than various delegate reqirements
ONNX_TO_SYNAP_TYPE: dict[str, DataType] = {
    'float32': DataType.FLOAT32,
    'float16': DataType.FLOAT16,
    'int32': DataType.INT32,
    'uint8': DataType.UINT8,
    'int16': DataType.INT16,
    'int8': DataType.INT8
}


class OnnxNetwork(Network):
    """
    Network class that was parsed from onnx
    """

    def __init__(self, model: ModelProto, model_path: Path) -> None:
        super().__init__()
        self.model_path = model_path
        self.model = model


def load_onnx(model_path: Path):
    """
    Converts a onnx model to
    """
    with open(model_path, "rb") as f:
        onnx_model = load(f)

    return shape_inference.infer_shapes(onnx_model)


def shape2list(shape):
    return list(getattr(d, 'dim_value', 0) for d in shape.dim)


def parse_onnx(model_path: Path) -> OnnxNetwork:
    """
    Parses a onnx file into a Network object
    If batch_size is dynamic the network is modified to use a batch_size of 1
    """
    onnx_model: ModelProto.ModelProto = load_onnx(model_path)

    # Before doing anything let's check that the onnx model itself is valid
    try:
        checker.check_model(onnx_model, full_check=True)
    except Exception as e:
        raise ConversionError("Invalid ONNX model:  " + str(e))

    input_shape_updated = False
    for input_node in onnx_model.graph.input:
        shape = shape2list(input_node.type.tensor_type.shape)
        if any(d <= 0 for d in shape[1:]):
            raise ConversionError(f"Dynamic tensor shape {shape} not supported in node {input_node.name}")
        if len(shape) > 0 and shape[0] <= 0:
            # Convert dynamic batch_size to batch_size of 1
            logger.info("Convert dynamic batch_size to 1 in input node: %s", input_node.name)
            shape[0] = 1
            onnx_model_utils.make_input_shape_fixed(onnx_model.graph, input_node.name, shape)
            input_shape_updated = True

    if input_shape_updated:
        # update the output shapes to make them fixed as well
        onnx_model_utils.fix_output_shapes(onnx_model)

        # Save the modified model and use it for further processing
        with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as tf:
            model_path = Path(tf.name)
            logger.debug("Modified onnx model saved to: %s", model_path)
        save(onnx_model, str(model_path))

    network = OnnxNetwork(onnx_model, model_path)

    # in the Network object we don't track constant tensors (i.e. parameters)
    initialized_value_names = ({x.name for x in onnx_model.graph.initializer} |
                               {x.name for x in onnx_model.graph.sparse_initializer})

    for idx, v in enumerate(
            itertools.chain(onnx_model.graph.value_info, onnx_model.graph.input, onnx_model.graph.output)):
        # we don't keep track of tensors that are not computed or network inputs
        if v.name in initialized_value_names:
            continue

        # values without name cannot be used as input/outputs of operations so we don't need to keep track of them
        if v.name is None or v.name == '':
            continue

        network.tensors_manager.add_tensor(TensorId(v.name),
                                          Tensor(v.name, shape2list(v.type.tensor_type.shape)))

    def filter_values(value_names):
        ret = []
        for value_name in value_names:
            # we don't keep track of tensors that are not computed or network inputs
            if value_name in initialized_value_names:
                continue
            ret.append(value_name)
        return ret

    for idx, node in enumerate(onnx_model.graph.node):
        inputs = filter_values(node.input)
        outputs = filter_values(node.output)

        if '' in inputs:
            # Remove anonymous inputs
            logger.info(f"Ignoring anonyous inputs in node: '{node.name}'")
            inputs = [i for i in inputs if i != '']

        network.operations_manager.add_operation(OperationId(str(idx)),
                                                 Operation(node.name, inputs, outputs))

    network.meta_manager.inputs = [x.name for x in onnx_model.graph.input]
    logger.debug("model inputs: %s", network.meta_manager.outputs)

    if None in network.meta_manager.inputs:
        raise Exception("Network with un-named inputs are not supported")

    network.meta_manager.outputs = [x.name for x in onnx_model.graph.output]
    logger.debug(f"model outputs: %s", network.meta_manager.outputs)

    if None in network.meta_manager.outputs:
        raise Exception("Network with un-named outputs are not supported")

    logger.debug(f"parsed model")

    return network


def write_onnx_slices(segmented_network: gs.SegmentedNetwork,
                        output_dir: Path, work_dir: Path) -> list[gs.SerializedSubgraph]:

    if not isinstance(segmented_network.original_network, OnnxNetwork):
        raise Exception("Cannot slice non-onnx network")

    network = segmented_network.original_network

    output_dir.mkdir(parents=True, exist_ok=True)

    tensor_data = {x.name: x
                   for x in itertools.chain(network.model.graph.input,
                                            network.model.graph.output,
                                            network.model.graph.value_info) if x.name is not None}

    slices = []
    for subgraph_idx, subgraph in segmented_network.subgraphs.items():
        logger.debug(f"slicing subgraph {subgraph_idx}")

        logger.debug(f"subgraph {subgraph_idx} inputs: {subgraph.inputs} outputs: {subgraph.outputs}")

        subgraph_dir_path = output_dir / str(subgraph_idx)
        subgraph_dir_path.mkdir(parents=True, exist_ok=True)
        out_model_path = subgraph_dir_path.joinpath(f"{subgraph_idx}.onnx").absolute()

        try:
            utils.extract_model(str(network.model_path), str(out_model_path), list(subgraph.inputs), list(subgraph.outputs))
        except onnx_cpp2py_export.checker.ValidationError as e:
            raise ConversionError("Invalid ONNX model: " + str(e))
        
        logger.debug(f"written subgraph to {out_model_path}")

        def tensor_id_to_subgraph_tensor(tn):
            tensor = tensor_data[tn]
            tensor_dtype = onnx_dtype_list[tensor.type.tensor_type.elem_type]
            if tensor_dtype not in ONNX_TO_SYNAP_TYPE:
                raise Exception(f"Type {tensor_dtype} of input {tn} is not supported")
            qi = None
            return SubgraphTensor(TensorId(tensor.name),
                                  GlobalTensorId(tensor.name),  # in onnx the two IDs match because we use names
                                                                # instead of indexes
                                  tn,
                                  TensorShape(shape2list(tensor.type.tensor_type.shape)),
                                  ONNX_TO_SYNAP_TYPE[tensor_dtype],
                                  qi)

        inputs = [tensor_id_to_subgraph_tensor(tensor_name) for tensor_name in subgraph.inputs]
        outputs = [tensor_id_to_subgraph_tensor(tensor_name) for tensor_name in subgraph.outputs]

        logger.debug(f"inputs: {inputs}, outputs: {outputs}")

        slices.append(
            gs.SerializedSubgraph(name=subgraph_idx, model_file=out_model_path, weights_file=None,
                                  inputs=inputs, outputs=outputs))

    return slices
