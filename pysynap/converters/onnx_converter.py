# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

import json
from pathlib import Path

from pysynap.converters import ConversionOptions, ConvertedModel
from pysynap.graph_segmenter import SerializedSubgraph, SubgraphTensor
from pysynap.meta import MetaInfo, DelegateInfo, Delegate, DataLayout
from pysynap.exceptions import ConversionError


# onnx quantization is add quantization layer to the model,
# all scale and zero point are initializer,
# seems runtime takes care of all quantization and dequantization


def convert(subgraph: SerializedSubgraph, subgraph_meta: MetaInfo, delegate: DelegateInfo,
            conversion_options: ConversionOptions, work_dir: Path):
    """
    Converts a serialized subgraph to onnx format suitable for
    execution with the cpu delegates. Currently,
    it only needs to create the meta.json file for the mode
    """

    if delegate.type == Delegate.CPU:
        delegate_string = "onnx cpu=1"
    elif delegate.type == Delegate.GPU:
        delegate_string = "onnx gpu=1"
    else:
        raise Exception("Delegate not supported")

    delegate_name = f"'{delegate.type.value}' for tflite"
    if subgraph_meta.quantization is not None:
        raise ConversionError(f"delegate {delegate_name} doesn't support model quantization")
    if subgraph_meta.security is not None:
        raise ConversionError(f"delegate {delegate_name} doesn't support model security")
    if subgraph_meta.layout not in [ DataLayout.DEFAULT, DataLayout.NCHW ]:
        raise ConversionError(f"delegate {delegate_name} doesn't support '{subgraph_meta.layout.value}' layout")

    delegate_string = " ".join([delegate_string, delegate.options])
    meta = {"Inputs": {}, "Outputs": {}, "secure": False, "delegate": delegate_string}

    def create_tensor_data(tensor: SubgraphTensor):
        tensor_data = {"shape": tensor.shape, "format": "nchw"}

        tensor_data["dtype"] = tensor.data_type.value

        if tensor.name is not None:
            tensor_data["name"] = tensor.name

        return tensor_data

    for idx, input_tensor in enumerate(subgraph.inputs):
        meta_input = subgraph_meta.inputs[idx]
        if meta_input.preproc and meta_input.preproc.preproc_type and meta_input.preproc.preproc_type != 'none':
            raise ConversionError(f"delegate {delegate_name} doesn't support preprocessing")
        new_tensor = create_tensor_data(input_tensor)
        new_tensor['data_format'] = meta_input.data_format or subgraph_meta.input_format
        new_tensor['security'] = meta_input.security

        if meta_input.means is not None:
            new_tensor['mean'] = meta_input.means

        if meta_input.scale is not None:
            new_tensor['scale'] = meta_input.scale

        meta['Inputs'][input_tensor.tensor_id] = new_tensor

    for idx, output_tensor in enumerate(subgraph.outputs):
        meta_output = subgraph_meta.outputs[idx]
        if subgraph_meta.dequantize_outputs or meta_output.dequantize:
            raise ConversionError(f"delegate {delegate_name} doesn't support output dequantization")
        new_tensor = create_tensor_data(output_tensor)
        new_tensor['data_format'] = meta_output.data_format or subgraph_meta.output_format
        new_tensor['security'] = meta_output.security
        meta['Outputs'][output_tensor.tensor_id] = new_tensor

    meta_path = subgraph.model_file.parent / (subgraph.model_file.stem + ".json")

    with open(meta_path, 'w') as fp:
        json.dump(meta, fp, indent=True)

    return ConvertedModel(model_file=subgraph.model_file, meta_file=meta_path)
