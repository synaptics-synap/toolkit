# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from random import Random

import networkx as nx

logger = logging.getLogger("synap.tests")

from pysynap.graph_segmenter import SegmentedNetwork, _organize_operations_in_subgraphs, SubgraphName, \
    Subgraph, TensorNode, OperationNode
from pysynap.meta import MetaInfo, Delegate, DelegateInfo, OutputInfo, DataFormat, InputInfo, DEFAULT_OUTPUT_SECURITY, \
    DEFAULT_INPUT_SECURITY
from pysynap.network import Network, Operation, OperationId, TensorId, Tensor, TensorShape, OperationPattern, \
    WILDCARD_OPERATION_PATTERN, parse_network, OperationName, TensorName, create_dataflow_graph, DataFlowGraph


@dataclass(frozen=True)
class SubgraphNode:
    name: SubgraphName


def generate_random_network(rnd: Random):
    """
    Creates a random network with operations with name "onX" and tensors with name "tnX" where X is a number.
    The operations have an id of the form "oX" and the tensors have an id of the form "tX" where X matches the name.
    """
    net = Network()

    # pick a random number of operations for the network
    net_count = rnd.randint(1, 50)

    # create a bunch of tensors that will be used as inputs to the network
    input_count = rnd.randint(1, 10)
    net.meta_manager.inputs = [TensorId("t" + str(x)) for x in range(input_count)]

    # this list will contain all the tensors that are available to be used as inputs to operations
    available_tensors = [*net.meta_manager.inputs]

    # keep track of all the tensors that are not used by any operation
    not_used_tensors = list(net.meta_manager.inputs)

    # add net_count operations to the network with random number of outputs and link them up to available tensors
    print(f"UUUUUUUUU the network has {net_count} operations")
    for idx in range(net_count):

        outputs_count = rnd.randint(1, 5)

        op_outputs = [TensorId("t" + str(x)) for x in range(len(available_tensors),
                                                            len(available_tensors) + outputs_count)]

        for x in op_outputs:
            if x not in not_used_tensors:
                not_used_tensors.append(x)

        op_inputs = rnd.sample(available_tensors, rnd.randint(1, min(len(available_tensors), 5)))

        for x in op_inputs:
            if x in not_used_tensors:
                not_used_tensors.remove(x)

        available_tensors += op_outputs

        net.operations_manager.add_operation(OperationId("o" + str(idx)),
                                             operation=Operation(name=OperationName("on" + str(idx)),
                                                                 inputs=op_inputs, outputs=op_outputs))

    # add the details for the tensors to the network
    for tensor in available_tensors:
        net.tensors_manager.add_tensor(tensor,
                                      Tensor(name=TensorName("tn" + tensor[1:]),
                                             shape=TensorShape([1,
                                                                rnd.randint(1, 100),
                                                                rnd.randint(1, 100),
                                                                rnd.randint(1, 100)])))

    # link any unused input to the first operation
    for tensor_id in net.meta_manager.inputs:
        if tensor_id in not_used_tensors:
            net.operations_manager.operations[OperationId("o0")].inputs.append(tensor_id)
            not_used_tensors.remove(tensor_id)

    # take all tensors that are computed but not used as network outputs, then add some other used tensors to the list
    candidate_outputs = set(not_used_tensors)
    candidate_outputs.update(rnd.sample(available_tensors, rnd.randint(1, len(available_tensors))))
    candidate_outputs = candidate_outputs - set(net.meta_manager.inputs)
    net.meta_manager.outputs = list(candidate_outputs)

    return net


def tensor_name_to_tensor_id(name: TensorName):
    return TensorId("t" + name[2:])


def tensor_id_to_tensor_name(tensor_id: TensorId):
    return TensorName("tn" + tensor_id[1:])


def create_meta(net: Network, rnd: Random):
    """
    Creates a metainfo object with random inputs, outputs and delegates compatible with the network passed as argument
    """

    meta = MetaInfo()

    # randomly assign nodes to the different delegates
    delegates = [DelegateInfo(Delegate.GPU, ''), DelegateInfo(Delegate.NPU, ''), DelegateInfo(Delegate.CPU, '')]

    meta.delegates[WILDCARD_OPERATION_PATTERN] = rnd.choice(delegates)

    for op in sorted(net.operations_manager.operations):
        if rnd.random() > 0.5:
            meta.delegates[OperationPattern(op)] = rnd.choice(delegates)

    # create a dataflow graph so that we can look at it to select some consistent choice of inputs and outputs
    graph = create_dataflow_graph(net)

    # randomly pick some output nodes from the nodes computed in the graph
    candidate_outputs = [x for x in graph.nodes if isinstance(x, TensorNode) and graph.in_degree(x) > 0]
    selected_outputs = list(rnd.sample(candidate_outputs, rnd.randint(1, len(candidate_outputs))))

    # randomly pick some input nodes, to do so we select them and disconnect them from their upstream operator
    for _ in range(rnd.randint(1, 10)):
        # find all the upstream nodes to the outputs (this can change when we disconnect the inputs we have chosen)
        upstream_nodes = list(itertools.chain.from_iterable(nx.ancestors(graph, x) for x in selected_outputs))

        # remove duplicates
        upstream_nodes = list(dict.fromkeys(upstream_nodes))

        graph.remove_nodes_from(graph.nodes - (set(upstream_nodes) | set(selected_outputs)))

        # the candidate inputs are any of the tensors in the upstream nodes that are not outputs
        available_inputs = [x for x in upstream_nodes if isinstance(x, TensorNode) and x not in selected_outputs]

        if len(available_inputs) == 0:
            break

        new_input = rnd.choice(available_inputs)

        # remove all the incoming edges to the tensor as it will be provided by the user
        edges = list(graph.in_edges(new_input))
        graph.remove_edges_from(edges)

    # select as inputs the tensors upstream from the outputs that have no operation generating them
    upstream_nodes = list(itertools.chain.from_iterable(nx.ancestors(graph, x) for x in selected_outputs))
    upstream_nodes = list(dict.fromkeys(upstream_nodes))

    graph.remove_nodes_from(graph.nodes - (set(upstream_nodes) | set(selected_outputs)))

    # Remove delegate specifications for layers not present in the graph
    graph_operation_ids = [x.id for x in graph.nodes if isinstance(x, OperationNode)]
    for operation_id in list(meta.delegates.keys()):
        if operation_id != WILDCARD_OPERATION_PATTERN and operation_id not in graph_operation_ids:
            del meta.delegates[operation_id]

    selected_inputs = [x for x in upstream_nodes if isinstance(x, TensorNode) and graph.in_degree(x) == 0]

    meta.inputs = [InputInfo(name=x.name, means=None, scale=None,
                             data_format=DataFormat(''), security=DEFAULT_INPUT_SECURITY, preproc=None,
                             tensor_h=None, tensor_w=None, shape=x.shape)
                   for x in [net.tensors_manager.tensors[y.id] for y in selected_inputs]]

    meta.outputs = [OutputInfo(name=x.name, dequantize=False,
                               data_format=DataFormat(''), security=DEFAULT_OUTPUT_SECURITY, shape=None)
                    for x in [net.tensors_manager.tensors[y.id] for y in selected_outputs]]

    return meta


def create_subgraph_dag(original_dag: DataFlowGraph, subgraph: Subgraph):
    """
    Creates a directed graph with nodes representing tensors and operations and edges representing the dataflow between
    them
    """
    subgraph_network = nx.DiGraph()

    for op in subgraph.ops:
        subgraph_network.add_node(OperationNode(op))

        for input_tensor in original_dag.predecessors(OperationNode(op)):
            subgraph_network.add_edge(input_tensor, OperationNode(op))

        for output_tensor in original_dag.successors(OperationNode(op)):
            subgraph_network.add_edge(OperationNode(op), output_tensor)

    return subgraph_network


def create_requested_network_dag(network: Network, meta: MetaInfo) -> DataFlowGraph:

    original_graph = create_dataflow_graph(network)

    # remove all the incoming edges to node
    for tensor in meta.inputs:
        edges = original_graph.in_edges(TensorNode(tensor_name_to_tensor_id(tensor.name)))
        original_graph.remove_edges_from(list(edges))

    # find all the requested output nodes
    if len(meta.outputs) > 0:
        output_nodes = [TensorNode(tensor_name_to_tensor_id(tensor.name)) for tensor in meta.outputs]
    else:
        output_nodes = [TensorNode(x) for x in network.meta_manager.outputs]

    # find all the nodes that are used by the output nodes
    used_nodes = set()

    for node in output_nodes:
        # find all the ancestors of the output tensor and add them to the set of used nodes
        used_nodes.update(nx.ancestors(original_graph, node))
        # add the output node itself
        used_nodes.add(node)

    # remove from the original graph all nodes that are not used
    original_graph.remove_nodes_from(set(original_graph.nodes) - used_nodes)

    for tensor in meta.inputs:
        assert original_graph.in_degree(TensorNode(tensor_name_to_tensor_id(tensor.name))) == 0

    return original_graph


def create_subgraphs_dag(segmented_net: SegmentedNetwork):
    """
    Creates a graph with nodes representing subgraphs and tensors and edges representing the dataflow between them
    """
    subgraph_graph = nx.DiGraph()

    # go over all the inputs and outputs of the subgraphs in the details and add them to the graph as tensor nodes
    for subgraph in segmented_net.subgraphs.values():
        for tensor in subgraph.inputs + subgraph.outputs:
            subgraph_graph.add_node(TensorNode(tensor))

    # go over all the subgraphs in the details and add them to the graph as subgraph nodes
    for name, subgraph in segmented_net.subgraphs.items():
        subgraph_graph.add_node(SubgraphNode(name))

        for input_tensor in subgraph.inputs:
            subgraph_graph.add_edge(TensorNode(input_tensor), SubgraphNode(name))

        for output_tensor in subgraph.outputs:
            subgraph_graph.add_edge(SubgraphNode(name), TensorNode(output_tensor))

    return subgraph_graph


def check_segmentation(net: Network, meta: MetaInfo, segmented_net: SegmentedNetwork):

    # create a DAG that represents the subset of the original network relationships requested by metainfo
    original_dag = create_requested_network_dag(net, meta)

    # check that the segments contain all the network operations requested by the user
    assert ({x.id for x in original_dag.nodes if isinstance(x, OperationNode)} ==
            set(itertools.chain.from_iterable(x.ops for x in segmented_net.subgraphs.values())))

    default_delegate = meta.delegates[WILDCARD_OPERATION_PATTERN]

    # check that all the nodes in the same subgraph are running on the correct delegate according to the meta
    for subgraph in segmented_net.subgraphs.values():
        for op in subgraph.ops:
            assert meta.delegates.get(OperationPattern(op), default_delegate) == subgraph.delegate

    # create a graph for each subgraph a DAG of the internal dataflow
    subgraph_dags = {}
    for subgraph_name, subgraph in segmented_net.subgraphs.items():
        subgraph_dags[subgraph_name] = create_subgraph_dag(original_dag, subgraph)

    # check that all the inputs of the subgraphs match what the operations expect
    # find all nodes in subgraph_network with in-degree 0
    for subgraph_name, subgraph in segmented_net.subgraphs.items():
        input_nodes = {x.id for x in subgraph_dags[subgraph_name].nodes
                       if subgraph_dags[subgraph_name].in_degree(x) == 0 and isinstance(x, TensorNode)}
        assert input_nodes == set(subgraph.inputs)

    # check that all the outputs of the subgraphs are tensors in the subgraph
    for subgraph_name, subgraph in segmented_net.subgraphs.items():
        assert {TensorNode(x) for x in subgraph.outputs} - set(subgraph_dags[subgraph_name].nodes) == set()

    # create a new graph by adding all the subgraphs of the segmented network
    # and check that after re-assembling the subgraph of the segmented network we obtain the original network
    # this ensures there are no missing or extra edges and nodes in the subgraph segmentation
    recomposed_subgraph = nx.DiGraph()

    for subgraph_name, subgraph in segmented_net.subgraphs.items():
        for node in subgraph_dags[subgraph_name].nodes:
            recomposed_subgraph.add_node(node)

        for u, v in subgraph_dags[subgraph_name].edges:
            recomposed_subgraph.add_edge(u, v)

    assert set(recomposed_subgraph.nodes) == set(original_dag.nodes)
    assert set(recomposed_subgraph.edges) == set(original_dag.edges)

    # check that the subgraphs execute in topological order create a topological order of nodes in the
    # original network, this ensures there are no data dependencies mistakes in the subgraph segmentation
    # FIXME: this is not an complete test probably because we don't check all the possible topological orders
    subgraph_graph = create_subgraphs_dag(segmented_net)

    # list of all the tensors computed so-far, initially this is the set of inputs specified in the meta if
    # specified otherwise use the network inputs
    if meta.inputs:
        computed_tensors = {tensor_name_to_tensor_id(x.name) for x in meta.inputs}
    else:
        computed_tensors = set(net.meta_manager.inputs)

    # go over all subgraphs in topological order (as the runtime will do)
    for subgraph_node in nx.topological_sort(subgraph_graph):
        # if we run into an operation node, check that it can be executed correctly
        if isinstance(subgraph_node, SubgraphNode):
            # go over all the operations of the subgraph in topological order (as the runtime will do)
            for node in nx.topological_sort(subgraph_dags[subgraph_node.name]):
                # in case the node is an operation check that the inputs are computed and add the outputs
                if isinstance(node, OperationNode):
                    for inp in net.operations_manager.operations[OperationId(node.id)].inputs:
                        assert inp in computed_tensors
                    for out in net.operations_manager.operations[OperationId(node.id)].outputs:
                        computed_tensors.add(out)


def test_organize_operations_in_subgraphs():
    rnd = Random(0)

    for _ in range(10):
        net = generate_random_network(rnd)

        meta = create_meta(net, rnd)

        default_delegate = DelegateInfo(Delegate.NPU, '')

        subgraph_details = _organize_operations_in_subgraphs(net, meta, default_delegate)

        check_segmentation(net, meta, subgraph_details)


def test_organize_operations_in_subgraphs_no_inputs_outputs():
    rnd = Random(1)

    for _ in range(10):
        net = generate_random_network(rnd)

        meta = create_meta(net, rnd)

        meta.inputs = []
        meta.outputs = []

        default_delegate = DelegateInfo(Delegate.NPU, '')

        subgraph_details = _organize_operations_in_subgraphs(net, meta, default_delegate)

        check_segmentation(net, meta, subgraph_details)


age_googlenet_path = ((Path(__file__)).parent.parent.parent.parent
                      / "models/image_classification/age/model/"
                        "age_googlenet_onnx/age_googlenet.onnx")

def test_organize_onnx(tmp_path):
    meta = MetaInfo()

    meta.delegates[WILDCARD_OPERATION_PATTERN] = DelegateInfo(Delegate.CPU, '')
    meta.delegates[OperationPattern("9")] = DelegateInfo(Delegate.GPU, '')

    net = parse_network(age_googlenet_path, tmp_path)

    default_delegate = DelegateInfo(Delegate.NPU, '')

    subgraph_details = _organize_operations_in_subgraphs(net, meta, default_delegate)

    assert len(subgraph_details.subgraphs) == 3
    assert set(subgraph_details.subgraphs[SubgraphName("subgraph_0")].ops) == {str(x) for x in range(9)}
    assert set(subgraph_details.subgraphs[SubgraphName("subgraph_1")].ops) == {"9"}

    assert set(subgraph_details.subgraphs[SubgraphName("subgraph_0")].inputs) == {"input"}
    assert set(subgraph_details.subgraphs[SubgraphName("subgraph_0")].outputs) == {"conv2/norm2_Y"}

    assert set(subgraph_details.subgraphs[SubgraphName("subgraph_1")].inputs) == {"conv2/norm2_Y"}
    assert set(subgraph_details.subgraphs[SubgraphName("subgraph_1")].outputs) == {"pool2/3x3_s2_Y"}

    assert set(subgraph_details.subgraphs[SubgraphName("subgraph_2")].inputs) == {"pool2/3x3_s2_Y"}
    assert set(subgraph_details.subgraphs[SubgraphName("subgraph_2")].outputs) == {"loss3/loss3_Y"}
