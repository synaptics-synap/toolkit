# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

from dataclasses import dataclass
from enum import Enum
from typing import NewType, TypeVar

import networkx as nx
from pathlib import Path
from .exceptions import ConversionError

"""
Optional human friendly name of a tensor
"""
TensorName = NewType('TensorName', str)

"""
Mandatory ID of a tensor
"""
TensorId = NewType('TensorId', str)

"""
Optional human friendly name of an operation
"""
OperationId = NewType('OperationId', str)

"""
Mandatory ID of an operation
"""
OperationName = NewType('OperationName', str)

"""
Shape of a tensor, which is a multidimensional array
"""
TensorShape = NewType('TensorShape', list[int])


class DataType(Enum):
    """
    Type of data stored in a tensor
    """
    BYTE = 'byte'
    UINT8 = 'uint8'
    INT8 = 'int8'
    UINT16 = 'uint16'
    INT16 = 'int16'
    INT32 = 'int32'
    UINT32 = 'uint32'
    FLOAT16 = 'float16'
    FLOAT32 = 'float32'


@dataclass
class Operation:
    """
    Represents a data processing node (a.k.a. layer) in a data processing network
    """

    name: OperationName | None
    inputs: list[TensorId]
    outputs: list[TensorId]


@dataclass
class Tensor:
    """
    Represents an input or output buffer of an Operation (a.k.a. buffer)
    """
    name: TensorName | None
    shape: TensorShape


class TensorsManager:
    """
    Manages the information about the tensors included in a network, these tensors are only the tensors that
    contains inputs to the network or outputs of operators. Notably, they don't include constant tensors that
    are inputs to the operations (e.g. weights, biases, etc.)
    """

    def __init__(self, network: 'Network'):
        self.network = network
        self.tensors_name_mapper: dict[TensorName | TensorId, TensorId] = {}
        self.tensors: dict[TensorId, Tensor] = {}

    def add_tensor(self, tensor_id: TensorId, tensor: Tensor):
        if tensor.name is not None:
            self.tensors_name_mapper[tensor.name] = tensor_id

            # FIXME: this is to be compatible with the current meta.yml values
            self.tensors_name_mapper[tensor.name.replace(":", "/")] = tensor_id
        self.tensors[tensor_id] = tensor

    def translate_name(self, tensor_name: TensorName) -> TensorId:
        """
        Finds the ID of a tensor with a given name
        """
        if not tensor_name in self.tensors_name_mapper:
            raise ConversionError(f"Tensor {tensor_name} not found in network.\nAvailable tensors:\n{list(self.tensors_name_mapper.keys())}")
        return self.tensors_name_mapper[tensor_name]


class OperationsManager:
    """
    Manages the information about the operations that compose a network
    """

    def __init__(self, network: 'Network'):
        self.network = network
        self.operations: dict[OperationId, Operation] = {}

    def add_operation(self, operation_id: OperationId, operation: Operation):
        self.operations[operation_id] = operation


class MetaManager:
    """
    Manages the metadata of a network
    """

    def __init__(self, network: 'Network'):
        self.network = network
        self.inputs: list[TensorId] = []
        self.outputs: list[TensorId] = []


class Network:
    """
    Represents a graph of operations that are consume some inputs and generates
    some outputs
    """
    def __init__(self) -> None:
        self.operations_manager: OperationsManager = OperationsManager(self)
        self.tensors_manager: TensorsManager = TensorsManager(self)
        self.meta_manager: MetaManager = MetaManager(self)


@dataclass(frozen=True, order=True)
class TensorNode:
    """
    Object used to represent a tensor in the dataflow graph
    """
    id: TensorId


@dataclass(frozen=True, order=True)
class OperationNode:
    """
    Object used to represent an operation in the dataflow graph
    """
    id: OperationId


DataFlowGraph = NewType('DataFlowGraph', nx.DiGraph)
"""
A DAG with tensor and operations as nodes and data dependencies as edges.

The tensors are represented by TensorNode objects and the operations by OperationNode objects.
"""


def create_dataflow_graph(network: Network) -> DataFlowGraph:
    """
    Creates a dataflow graph from a network
    """

    graph = nx.DiGraph()

    for tensor_id, tensor in network.tensors_manager.tensors.items():
        graph.add_node(TensorNode(tensor_id))

    for operation_id, operation in network.operations_manager.operations.items():
        graph.add_node(OperationNode(operation_id))

        for tensor_id in operation.inputs:
            graph.add_edge(TensorNode(tensor_id), OperationNode(operation_id))

        for tensor_id in operation.outputs:
            graph.add_edge(OperationNode(operation_id), TensorNode(tensor_id))

    return DataFlowGraph(graph)


OperationPattern = NewType('OperationPattern', str)
"""
Operator patterns are used to match one or more operations in a network
"""

WILDCARD_OPERATION_PATTERN = OperationPattern('*')
"""
Matches any operation in the network
"""

DESCENDANTS_PATTERN_SUFFIX = OperationPattern('...')
"""
Matches all the operations that are reachable from a given operation in the network graph
"""

INPUTS_PATTERN = OperationPattern('INPUTS')
"""
Matches all the operations that are inputs of the network
"""


T = TypeVar('T')


def match_patterns(graph: DataFlowGraph, patterns: dict[OperationPattern, T], default_value: T) -> dict[OperationId, T]:
    """
    For each operation in the network find the pattern that matches it and return a dictionary with the
    value associated with the most specific pattern that matches each operation.

    The patterns are applied in the following order:
        - default rule (WILDCARD_OPERATION_PATTERN)
        - node specific and descendants rules (patterns ending with DESCENDANTS_PATTERN_SUFFIX)

    If not pattern matches a node then the default value specified as parameter is used.

    The patterns match operations as follows:
        - default rule matches all the operations
        - descendants rules match all the operations that are reachable in the network graph from the operation with the
          id specified in the pattern before the DESCENDANTS_PATTERN_SUFFIX
        - node specific rules match the operation with the id specified in the pattern
    """

    default_value = patterns.get(WILDCARD_OPERATION_PATTERN, default_value)

    matches: dict[OperationId, T] = {}

    # first match the default pattern to all operations
    for operation_id in [x.id for x in graph.nodes if isinstance(x, OperationNode)]:
        matches[operation_id] = default_value

    # then try to match descendants suffix patterns
    for pattern, pattern_value in patterns.items():
        if pattern == INPUTS_PATTERN:
            raise ConversionError(f"{INPUTS_PATTERN} pattern not supported in delegate assignment")
        if pattern == WILDCARD_OPERATION_PATTERN:
            continue
        if pattern.endswith(DESCENDANTS_PATTERN_SUFFIX):
            root_node_id = pattern[:-len(DESCENDANTS_PATTERN_SUFFIX)]
            if root_node_id not in matches:
                raise ConversionError(f"Layer ID '{root_node_id}' not in: {matches.keys()}")
            descendants = nx.descendants(graph, OperationNode(root_node_id))
            for operation_id in [x.id for x in descendants if isinstance(x, OperationNode)]:
                matches[operation_id] = pattern_value

        else:
            if pattern not in matches:
                raise ConversionError(f"Layer ID '{pattern}' not in: {matches.keys()}")
            matches[OperationId(pattern)] = pattern_value

    return matches


def parse_network(model_path: Path, work_dir: Path):

    if model_path.suffix == '.tflite':
        # import here to avoid circular dependencies
        from .utils.format_handlers.tflite_parser import parse_tflite

        network = parse_tflite(model_path, work_dir)

    elif model_path.suffix == '.onnx':
        from .utils.format_handlers.onnx_parser import parse_onnx
        network = parse_onnx(model_path)

    else:
        raise Exception("Unsupported format " + model_path.suffix)

    return network
