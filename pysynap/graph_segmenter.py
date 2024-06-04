# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

import logging
from dataclasses import dataclass
from typing import NewType

import networkx as nx

from pathlib import Path

from .exceptions import ConversionError
from .meta import Delegate, MetaInfo, DelegateInfo, InputInfo, DEFAULT_INPUT_SECURITY, OutputInfo, DEFAULT_OUTPUT_SECURITY, \
    UNKNOWN_DATA_FORMAT
from .network import Network, OperationId, TensorShape, TensorId, parse_network, match_patterns, \
    WILDCARD_OPERATION_PATTERN, DataType, create_dataflow_graph, TensorNode, OperationNode, DataFlowGraph

logger = logging.getLogger("synap.graph_segmenter")

GlobalTensorId = NewType('GlobalTensorId', str)
"""
Tensor identifier that is unique across all subgraph of a given network
"""

NodeLabel = NewType("NodeLabel", tuple[bool, Delegate])
"""
Label used during segmentation of a network to identify
"""

SubgraphName = NewType("SubgraphName", str)
"""
Name of a subgraph of a segmented network
"""


@dataclass
class QuantizationParameters:
    """
    Information required to convert entries of a tensor to/from their floating point representation
    """
    scale: float
    zero_point: float


@dataclass
class SubgraphTensor:
    """
    Represents the input or output tensor of a subgraph
    """
    tensor_id: TensorId
    """
    Id of the tensor in the subgraph after it it has been splitted from the original network
    """

    global_tensor_id: GlobalTensorId
    """
    Id of the tensor in the original network before splitting
    """

    name: str | None
    """
    Optional name of the tensor
    """

    shape: TensorShape
    """
    Shape of the tensor, which is a multidimensional array
    """

    data_type: DataType
    """
    Type of each entry of the tensor
    """

    quantization_params: QuantizationParameters | None
    """
    Details used to map the values stored in the tensor to the actual floating point value it represents
    """


@dataclass
class SerializedSubgraph:
    """
    Tracks information about a subgraph that was serialized to the specified file.
    """
    name: SubgraphName
    model_file: Path
    weights_file: Path
    inputs: list[SubgraphTensor]
    outputs: list[SubgraphTensor]


@dataclass
class SerializedSegmentedNetwork:
    """
    Tracks information about a network that has been segmented and serialized to files.

    This information is used during the conversion process after the original network
    has been split into files. The conversion process takes this information, converts
    each subgraph to the format suitable for the target delegate of each subgraph and
    create the final bundle file.
    """
    subgraphs: dict[SubgraphName, SerializedSubgraph]
    target_delegates: dict[SubgraphName, DelegateInfo]
    subgraph_metas: dict[SubgraphName, MetaInfo]
    inputs: list[GlobalTensorId]
    outputs: list[GlobalTensorId]


@dataclass
class Subgraph:
    """
    A subset of operations of a network that need to be executed with a given delegate
    """

    ops: list[OperationId]
    """
    Operators that make up the subgraph
    """

    delegate: Delegate
    """
    Delegate that will execute the graph
    """

    computed_tensors: set[TensorId]
    """
    Set of all tensors that are computed by the operations in this graph (does not include the input tensors)
    """

    inputs: set[TensorId]
    """
    List of all the tensors that are required as input to compute this subgraph
    """

    outputs: set[TensorId]
    """
    List of all the tensors that are generated as output by this subgraph, these are tensors that are used
    either by other subgraphs or are outputs of the overall segmented network.
    """


@dataclass
class SegmentedNetwork:
    """
    Describes a network partitioned in multiple subgraphs
    """
    subgraphs: dict[SubgraphName, Subgraph]
    original_network: Network
    inputs: list[TensorId]
    outputs: list[TensorId]


def _get_requested_input_tensors(meta: MetaInfo, network: Network) -> list[TensorId]:
    """
    Returns the id of the input tensors of network the user requested (using the inputs section of the meta info)
    """
    if meta.input_names_str(False):
        # meta declared inputs by specifying their names
        input_tensors = [network.tensors_manager.translate_name(x.name) for x in meta.inputs]
    else:
        # the user didn't specify any input so we take all the inputs of the original network
        input_tensors = network.meta_manager.inputs

    if len(meta.inputs) > len(input_tensors):
        raise ConversionError(f"Model has {len(input_tensors)} inputs but {len(meta.inputs)} inputs specified in meta file")

    logger.debug("converted network input tensors: %s", input_tensors)

    return input_tensors


def _get_requested_output_tensors(meta: MetaInfo, network: Network) -> list[TensorId]:
    """
    Returns the id of the output tensors that network the user requested (using the outputs section of the meta info)
    """
    if meta.output_names_str(False):
        # meta declared outputs by specifying their names
        output_tensors = [network.tensors_manager.translate_name(x.name) for x in meta.outputs]
    else:
        # the user didn't specify any output so we take all the outputs of the original network
        output_tensors = network.meta_manager.outputs

    if len(meta.outputs) > len(output_tensors):
        raise ConversionError(f"Model has {len(output_tensors)} outputs but {len(meta.outputs)} outputs specified in meta file")

    logger.debug("converted network output tensors: %s", output_tensors)

    return output_tensors


def _create_subgraph_meta(network: Network, subgraph: SerializedSubgraph,
                          delegate: Delegate, meta: MetaInfo) -> MetaInfo:
    """
    Takes the meta info of the overall conversion tasks and creates a meta info that describes the conversion
    of a single subgraph.
    """
    subgraph_meta = MetaInfo()

    subgraph_meta.network_format = meta.network_format
    subgraph_meta.layout = meta.layout
    subgraph_meta.input_format = meta.input_format
    subgraph_meta.output_format = meta.output_format
    subgraph_meta.optimize = meta.optimize
    subgraph_meta.inputs = [_find_input_info_for_tensor(x.global_tensor_id, network, meta) for x in subgraph.inputs]
    subgraph_meta.outputs = [_find_output_info_for_tensor(x.global_tensor_id, network, meta) for x in subgraph.outputs]
    subgraph_meta.security = meta.security
    subgraph_meta.quantization = meta.quantization
    subgraph_meta.delegates = {WILDCARD_OPERATION_PATTERN: delegate}

    return subgraph_meta


def _find_input_info_for_tensor(tensor_id: GlobalTensorId, network: Network, meta: MetaInfo):
    """
    Finds the InputInfo specified in the MetaInfo if it is specified by the user, otherwise generate a new default one.
    """

    # we will use the tensor id only in the context of the original network, so we can cast it to a TensorId
    tensor_id = TensorId(tensor_id)

    # when the meta info contains inputs, the input info for this tensor may be specified there
    if meta.inputs:

        # the meta info may be specified in an input that specifies the tensor name
        tensor_name = network.tensors_manager.tensors[tensor_id].name

        # the tensor name
        if tensor_name is not None:

            input_infos = [x for x in meta.inputs if x.name == tensor_name]

            if len(input_infos) > 0:
                return input_infos[0]

        # we didn't find the tensor by name, maybe it is specified by index, but in this case it
        # must also be an input of the original network
        if tensor_id in network.meta_manager.inputs:
            network_input_index = network.meta_manager.inputs.index(tensor_id)

            # make sure the user specified this input as an input (maybe they specified less inputs)
            if len(meta.inputs) > network_input_index:

                # make sure the user didn't substitute the input with another named tensor
                if meta.inputs[network_input_index].name is None:
                    return meta.inputs[network_input_index]

    # at this point we know the original meta info doesn't contain any InputInfo about the tensor so we
    # know we must generate it

    return InputInfo(name=None, means=None, scale=None, data_format=UNKNOWN_DATA_FORMAT,
                     security=DEFAULT_INPUT_SECURITY, preproc=None, tensor_w=None, tensor_h=None, shape=None)


def _find_output_info_for_tensor(tensor_id: GlobalTensorId, network: Network, meta: MetaInfo):
    """
    Finds the OutputInfo specified in the MetaInfo if it is specified by the user, otherwise generate a new default one.
    """

    # we will use the tensor id only in the context of the original network, so we can cast it to a TensorId
    tensor_id = TensorId(tensor_id)

    # when the meta info contains outputs, the output info for this tensor may be specified there
    if meta.outputs:

        # the meta info may be specified in an output that specifies the tensor name
        tensor_name = network.tensors_manager.tensors[tensor_id].name

        # the tensor name
        if tensor_name is not None:

            output_infos = [x for x in meta.outputs if x.name == tensor_name]

            if len(output_infos) > 0:
                return output_infos[0]

        # we didn't find the tensor by name, maybe it is specified by index, but in this case it
        # must also be an output of the original network
        if tensor_id in network.meta_manager.outputs:
            network_output_index = network.meta_manager.outputs.index(tensor_id)

            # make sure the user specified this output as an output (maybe they specified fewer outputs)
            if len(meta.outputs) > network_output_index:

                # make sure the user didn't substitute the outputs with another named tensor
                if meta.outputs[network_output_index].name is None:
                    return meta.outputs[network_output_index]

    # at this point we know the original meta info doesn't contain any InputInfo about the tensor, so we
    # know we must generate it

    return OutputInfo(name=None, dequantize=False, data_format=UNKNOWN_DATA_FORMAT, security=DEFAULT_OUTPUT_SECURITY, shape=None)


def _write_subgraphs(network_path: Path, segmented_network: SegmentedNetwork,
                     output_dir: Path, work_dir: Path) -> list[SerializedSubgraph]:
    """
    Write to disk the model files for each subgraph of a segmented network.
    """

    if network_path.suffix == ".tflite":
        # import here to avoid circular dependencies
        from .utils.format_handlers.tflite_parser import write_tflite_slices
        return write_tflite_slices(segmented_network, output_dir, work_dir)

    elif network_path.suffix == ".onnx":
        from .utils.format_handlers.onnx_parser import write_onnx_slices
        return write_onnx_slices(segmented_network, output_dir, work_dir)
    else:
        raise Exception(f"Cannot segment network of format {network_path.suffix}")


def segment(network_path: Path, meta: MetaInfo, default_delegate: DelegateInfo,
            output_path: Path, work_dir: Path) -> SerializedSegmentedNetwork:
    """
    Segment a given network model using the delegate information specified in the meta info, write out
    the segments to the output_path and returns the metadata describing the segmented network along with
    the meta conversion information.
    """

    # create an internal representation of the network
    parsing_work_dir = work_dir / 'parsing'
    parsing_work_dir.mkdir()
    network = parse_network(model_path=network_path, work_dir=parsing_work_dir)

    # find a way to collect subgraph nodes so that each subgraph is executed by only one delegate
    # this returns a list of operations for each subgraph and the delegate that must run them
    subgraph_details = _organize_operations_in_subgraphs(network, meta, default_delegate)

    # extract from the original model the subgraphs and write them the output path
    splitting_work_dir = work_dir / 'splitting'
    splitting_work_dir.mkdir()
    serialized_subgraphs = _write_subgraphs(network_path, subgraph_details, output_path, splitting_work_dir)

    # compute the meta information for each subgraph
    subgraphs_meta = {}
    for subgraph in serialized_subgraphs:
        subgraphs_meta[subgraph.name] = _create_subgraph_meta(network, subgraph,
                                                              subgraph_details.subgraphs[subgraph.name].delegate, meta)

    return SerializedSegmentedNetwork(subgraphs={x.name: x for x in serialized_subgraphs},
                                      target_delegates={x: y.delegate for x, y in subgraph_details.subgraphs.items()},
                                      inputs=[GlobalTensorId(x) for x in subgraph_details.inputs],
                                      outputs=[GlobalTensorId(x) for x in subgraph_details.outputs],
                                      subgraph_metas=subgraphs_meta)


def _extract_requested_network(network: Network, input_tensors: list[TensorId], output_tensors: list[TensorId]) \
        -> DataFlowGraph:
    """
    Creates a DAG that represents the tensors and operations the user is interested in from the original network.
    This is based on the information provided in the meta info (inputs and outputs section).
    """

    graph = create_dataflow_graph(network)

    # disconnect all inputs from the operation that generates them as the user wants to provide them as inputs
    for input_tensor in input_tensors:
        graph.remove_edges_from([(x, TensorNode(input_tensor)) for x in graph.predecessors(TensorNode(input_tensor))])

    # find all the nodes that are required to compute the outputs so that we can discard all nodes that are
    # not needed to compute the outputs the user is interested in
    needed_nodes = {TensorNode(t) for t in output_tensors}
    for output_tensor in output_tensors:
        needed_nodes.update(nx.ancestors(graph, TensorNode(output_tensor)))

    # remove all the nodes that are not needed to compute the outputs from the graph
    graph.remove_nodes_from(set(graph.nodes) - needed_nodes)

    # find all the inputs required to compute the outputs (after we have disconnected the requested inputs from
    # their source operation if necessary
    effective_inputs = {x for x in graph.nodes if graph.in_degree(x) == 0 and isinstance(x, TensorNode)}

    # create a list of nodes for all the inputs that have been requested by the user
    requested_input_nodes = {TensorNode(x) for x in input_tensors}

    # check that there are no requested input nodes that are unused as we don't want to include them in
    # the converted network
    unused_inputs = requested_input_nodes - effective_inputs

    if len(unused_inputs) > 0:
        unused_input_names = [network.tensors_manager.tensors[x.id].name for x in unused_inputs]
        raise Exception(f"The inputs {unused_input_names} are not necessary to compute the network outputs, "
                        "please exclude them by modifying the meta info")

    # check that all the input tensors required to compute the network have been requested
    missing_inputs = effective_inputs - requested_input_nodes
    if len(missing_inputs) > 0:
        missing_nodes_names = [network.tensors_manager.tensors[x.id].name for x in missing_inputs]
        raise Exception(f"The inputs {missing_nodes_names} are required to compute the network outputs but "
                        "are not included in the network, please add them to the meta info inputs section")

    for input_id in input_tensors:
        if graph.out_degree(TensorNode(input_id)) == 0:
            raise Exception(f"The input tensor {network.tensors_manager.tensors[input_id].name} is not used"
                            " by any operation in the network, please remove it from the network using the"
                            " meta info inputs sections")

    for output_id in output_tensors:
        if graph.in_degree(TensorNode(output_id)) == 0:
            raise Exception(f"The output tensor {network.tensors_manager.tensors[output_id].name} is not generated"
                            " by any operation in the network, please remove it from the network using the"
                            " meta info outputs sections")

    return graph


OperationsGraph = NewType('OperationsGraph', nx.DiGraph)
"""
Graph representing the operations in a network and the data dependencies between them. Nodes are operations 
(represented by OperationNode objects) and edges from X to Y represent the fact that operation X generates data
that is a direct input by operation Y.
"""


def _create_operations_graph(dataflow_graph: DataFlowGraph) -> OperationsGraph:
    """
    Creates an operation graph from a network represented by a data flow graph.
    """

    operations_graph = nx.DiGraph()

    operations_graph.add_nodes_from([x for x in dataflow_graph.nodes if isinstance(x, OperationNode)])

    for node in operations_graph.nodes:
        for tensor_successor in dataflow_graph.successors(node):
            for operation_successor in dataflow_graph.successors(tensor_successor):
                operations_graph.add_edge(node, operation_successor)

    return OperationsGraph(operations_graph)


def _organize_operations_in_subgraphs(network: Network, meta: MetaInfo, default_delegate: DelegateInfo) -> SegmentedNetwork:
    """
    Computes the SegmentedNetwork from the given network using the information derived from the meta.yml
    file found in the info object
    """

    # find the input and output tensors requested by the user (based on inputs/outputs in the MetaInfo if
    # that is present)
    input_tensors = _get_requested_input_tensors(meta, network)
    output_tensors = _get_requested_output_tensors(meta, network)

    # extract the graph that the user has requests
    requested_dataflow_graph = _extract_requested_network(network, input_tensors, output_tensors)

    # create a graph that only represents the data dependencies between operations in the network
    graph = _create_operations_graph(requested_dataflow_graph)

    # graphs should not be cyclic, but we check just in case to avoid ending up in an infinite loop
    if not nx.is_directed_acyclic_graph(graph):
        raise Exception("Input graph is cyclic")

    # labels for each node in the graph, these are the names of the delegates that will be used to compute the node
    node_labels = match_patterns(requested_dataflow_graph, meta.delegates, default_delegate)

    # nodes for which all the predecessor operations have been added to the computation order
    # initially this list contains all nodes with 0 in-degree (i.e. no predecessors)
    # FIXME: we could be more time efficient if we would keep this list segmented by delegate
    ready = sorted([node_idx for node_idx in graph.nodes if graph.in_degree(node_idx) == 0], key=lambda x: x.id)

    # number of predecessors we need to compute to be able to execute each node
    missing_predecessors = {x: graph.in_degree(x) for x in graph.nodes}

    # all subgraphs computed so far in execution order
    subgraphs: list[Subgraph] = []

    # iterate until there are nodes that are ready, if the graph is acyclic this will be true until all nodes
    # have been added to the computation
    while len(ready) > 0:

        # candidates to extend the current subgraph, these are nodes that have all their input available
        # we initialize this to the current ready list and we will add new nodes to it as we process the
        # list
        to_check = list(ready)

        # nodes that are ready but that we cannot add to the current subgraph, we will process them at the
        # next iteration
        next_ready = []

        # pick the first node to add to the next subgraph
        node = to_check.pop()

        # create the next subgraph and set its delegate
        current_subgraph = Subgraph(ops=[], delegate=node_labels[node.id],
                                    computed_tensors=set(), inputs=set(), outputs=set())

        logger.info("creating subgraph with delegate %s", current_subgraph.delegate)

        # iterate over the to_check list to add as many nodes as possible to the current subgraph
        while True:

            # if the node being processed is not computed with the current delegate add it to the list of
            # nodes that will be scanned for the next subgraph
            if node_labels[node.id] != current_subgraph.delegate:
                next_ready.append(node)
            else:
                # add the node to the current subgraph
                current_subgraph.ops.append(node.id)

                # add the output tensors of the node to the tensors of the subgraph
                current_subgraph.computed_tensors.update({x.id for x in requested_dataflow_graph.successors(node)})

                # add the inputs of the node that are not generated by operations of this subgraph to the inputs of the
                # subgraph
                current_subgraph.inputs.update({x.id for x in requested_dataflow_graph.predecessors(node)
                                                if x.id not in current_subgraph.computed_tensors})

                # decrease by one the missing predecessors from all the successors of the current node
                for successor in graph.successors(node):
                    m = missing_predecessors[successor]

                    # by construction, we should never encounter a node with 0 missing predecessors
                    assert m > 0

                    # update the number of missing predecessors for the successor
                    missing_predecessors[successor] = m - 1

                    # we are ready to compute this node, add it to the nodes to be checked
                    if m == 1:
                        to_check.append(successor)

            # we don't have any other node that has not been checked, so we need to finish the current graph
            if len(to_check) == 0:
                break

            # check the next node that is a valid candidate to be added to the current subgraph
            node = to_check.pop()

        # add the current subgraph to the list execution list
        subgraphs.append(current_subgraph)

        # update the ready list with the nodes that are ready to be processed
        ready = next_ready

    # find the inputs and outputs of the overall network
    logger.debug("overall network inputs: %s", input_tensors)
    logger.debug("overall network outputs: %s", output_tensors)

    # find the list of all the tensors that are used as inputs to any of the subgraphs
    all_inputs = set()
    for subgraph in subgraphs:
        all_inputs.update(subgraph.inputs)

    # create a list of all cross subgraph tensors, tensors that are either an input subgraph or
    # an output of the overall network
    cross_subgraph_tensors = all_inputs | set(output_tensors)

    # update the list of outputs of each subgraph by finding which of their tensors is a cross subgraph tensor
    for subgraph in subgraphs:
        subgraph.outputs = subgraph.computed_tensors.intersection(cross_subgraph_tensors)
        # Sort the output tensors to ensure deterministic order
        # The actual order doesn't really matter because internally subgraph tensors are always
        # referenced by their id
        subgraph.outputs = sorted(list(subgraph.outputs))
        subgraph.inputs = sorted(list(subgraph.inputs))

        logger.debug("created subgraph with delegate %s", subgraph.delegate)
        logger.debug("nodes %s", subgraph.ops)
        logger.debug("inputs %s", subgraph.inputs)
        logger.debug("outputs %s", subgraph.outputs)

    return SegmentedNetwork(subgraphs={f"subgraph_{idx}": subgraph for idx, subgraph in enumerate(subgraphs)},
                            original_network=network, inputs=input_tensors, outputs=output_tensors)
