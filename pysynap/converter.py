# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

import pysynap.converters
import logging
import shutil
import os
import json
from ruamel.yaml import YAML
from .bundler import Bundler
from .converters import ConversionOptions, TargetSOC
from .graph_segmenter import segment, SerializedSubgraph, SubgraphName, GlobalTensorId, SubgraphTensor, SerializedSegmentedNetwork
from .meta import *
from .network import TensorId

logger = logging.getLogger("synap")

def get_tensor(ssn: SerializedSegmentedNetwork, global_tensor_id: GlobalTensorId):
    for subgraph in ssn.subgraphs.values():
        for tensor in subgraph.inputs + subgraph.outputs:
            if tensor.global_tensor_id == global_tensor_id:
                return tensor
    return None

def tensor_to_str(tensor: SubgraphTensor):
    s = f'  - name: {tensor.name}\n    shape: {tensor.shape}\n    type: {tensor.data_type.value}'
    if tensor.quantization_params:
        s += ' quantized'
    return s

def tensor_to_info(tensor: dict):
    t = {
        'name':tensor['name'],
        'shape':tensor['shape'],
        'layout':tensor.get('format'),
        'format':tensor['data_format'],
        'type':tensor.get('dtype')
    }
    quantize = tensor.get('quantize')
    if quantize:
        t['type'] = quantize['qtype'] 
        if t['type'] == 'u8':
            t['type'] = 'uint8'
        qi = { 'scheme': tensor['quantizer'] }
        if 'scale' in quantize:
            qi['scale'] = quantize['scale']
        if 'zero_point' in quantize:
            qi['zero_point'] = quantize['zero_point']
        if 'fl' in quantize:
            qi['fractional_length'] = quantize['fl']
        t['quantization'] = qi
    if 'security' in tensor:
        t['security'] = tensor['security']
    return t


class HeterogeneousConverter:
    """
    Converts a model into according to a MetaInfo conversion configuration.
    """

    def __init__(self, conversion_options: ConversionOptions):
        self.conversion_options = conversion_options

    def _available_delegates_for_target(self) -> list[Delegate]:
        """
        Returns the list of delegates available for the target SoC.
        The first in the list is the default delegate.
        """
        if self.conversion_options.target in [TargetSOC.VS640, TargetSOC.VS680, TargetSOC.SL1640, TargetSOC.SL1680]:
            return [ Delegate.NPU, Delegate.GPU, Delegate.CPU ]
        elif self.conversion_options.target in [TargetSOC.DVF120, TargetSOC.SL1620]:
            return [ Delegate.CPU, Delegate.GPU ]
        return [ Delegate.CPU ]

    def convert(self, model_path: Path, weights_path: Path, meta: MetaInfo, output_path: Path, work_dir: Path):
        """
        Depending on the delegates specified in @param:meta,
        the model specified at @param:model_path is partitioned
        into multiple subgraphs and written to delegate specific
        formats to dir specified @param:output_dir.
        """

        subgraph_details_outdir = work_dir / 'subgraphs'
        subgraph_details_outdir.mkdir()

        subgraph_details_workdir = work_dir / 'subgraphs_workdir'
        subgraph_details_workdir.mkdir()

        bundler = Bundler()
        listing_files = {}

        # Replace 'default' delegate with the actual default delegate for the selected target
        target_delegates = self._available_delegates_for_target()
        default_target_delegate = target_delegates[0]
        logger.info("Default delegate for target SoC: " + default_target_delegate.value)
        for delegate_name, delegate_info in meta.delegates.items():
            if delegate_info.type == Delegate.DEFAULT:
                delegate_info.type = default_target_delegate
            elif delegate_info.type not in target_delegates:
                raise ConversionError(f"Delegate '{delegate_info.type.value}' not supported for target SoC: {self.conversion_options.target.value}")

        if not all(d.type == Delegate.NPU for d in meta.delegates.values()):

            subgraph_details = segment(network_path=model_path, meta=meta,
                                       default_delegate=DelegateInfo(default_target_delegate, ''),
                                       output_path=subgraph_details_outdir,
                                       work_dir=subgraph_details_outdir)

            # convert all models that are not in a format supported by the target delegate
            for subgraph_name, subgraph in subgraph_details.subgraphs.items():

                target_delegate = subgraph_details.target_delegates[subgraph_name]
                meta = subgraph_details.subgraph_metas[subgraph_name]

                subgraph_conversion_workdir = work_dir / 'conversion' / subgraph_name
                subgraph_conversion_workdir.mkdir(parents=True)

                # Normally output tensor shapes are extracted from the model, but in some
                # cases this information is not available (eg.TFLite_Detection_PostProcess layer).
                # For this reason it is possible to specify the output tensor shape explicitly in the metafile.
                # In this case it must match the actual tensor shape if available.
                for i, t in enumerate(subgraph.outputs):
                    if meta.outputs[i].shape:
                        if not t.shape or t.shape == meta.outputs[i].shape:
                            t.shape = meta.outputs[i].shape
                        else:
                            raise ConversionError(f"Output tensor {t.name} shape specified in conversion metafile {meta.outputs[i].shape} does not match "
                                             f"actual tensor shape: {t.shape}")

                if target_delegate.type == Delegate.NPU:
                    # Force floating point in/out tensors of the subgraph to be float32 if they are not the
                    # actual in/out of the network. This is because other delegates do not support
                    # float16 data type which is used by default by the NPU
                    for i, t in enumerate(subgraph.inputs):
                        if t.data_type == DataType.FLOAT32 and t.global_tensor_id not in subgraph_details.inputs:
                            meta.inputs[i].preproc = PreprocInfo('float32', None, None)
                    for i, t in enumerate(subgraph.outputs):
                        if t.data_type == DataType.FLOAT32 and t.global_tensor_id not in subgraph_details.outputs:
                            meta.outputs[i].dequantize = True

                converted_model = pysynap.converters.convert(subgraph,
                                                             subgraph_details.subgraph_metas[subgraph_name],
                                                             target_delegate,
                                                             self.conversion_options,
                                                             subgraph_conversion_workdir)

                bundler.add_subgraph(subgraph_name=subgraph_name,
                                     input_tensors=subgraph.inputs,
                                     output_tensors=subgraph.outputs,
                                     model_filename=converted_model.model_file,
                                     meta_filename=converted_model.meta_file,
                                     delegate=target_delegate)

                if converted_model.listing_files:
                    listing_files[subgraph_name] = converted_model.listing_files

            bundler.inputs = subgraph_details.inputs
            bundler.outputs = subgraph_details.outputs

        else:
            # Convert directly with NPU EBG delegate without splitting in subgraphs
            # This allows to support features such as .pb and .caffe model import that are not
            # supported in the general case
            logger.info("Converting model with NPU EBG delegate without splitting")
            subgraph = SerializedSubgraph(SubgraphName('subgraph_0'), model_path, weights_path, [], [])

            from pysynap.converters.acuity_converter import convert
            converted_model = convert(subgraph, meta, Delegate.NPU, self.conversion_options,
                                      subgraph_details_workdir, subgraph_details_workdir)

            # Tensor attributes not actually used, so use dummy values
            dummy_tensor = lambda ix: SubgraphTensor(TensorId(str(ix)), GlobalTensorId(str(ix)),
                                                     '', TensorShape([]), DataType.BYTE, None)

            subgraph.inputs = [dummy_tensor(ix) for ix in range(converted_model.input_count)]
            subgraph.outputs = [dummy_tensor(ix + len(subgraph.inputs)) for ix in range(converted_model.output_count)]

            bundler.add_subgraph(subgraph_name=subgraph.name,
                                 input_tensors=subgraph.inputs,
                                 output_tensors=subgraph.outputs,
                                 model_filename=converted_model.model_file,
                                 meta_filename=converted_model.meta_file,
                                 delegate=Delegate.NPU)

            if converted_model.listing_files:
                listing_files[subgraph.name] = converted_model.listing_files

            bundler.inputs = [t.tensor_id for t in subgraph.inputs]
            bundler.outputs = [t.tensor_id for t in subgraph.outputs]
            subgraph_details = SerializedSegmentedNetwork(
                {subgraph.name: subgraph},
                {subgraph.name: DelegateInfo(Delegate.NPU, '')},
                None,
                bundler.inputs,
                bundler.outputs
            )

        bundler.create_bundle(output_path)
        self._generate_listing_files(output_path, subgraph_details, listing_files, bundler)


    @staticmethod
    def _generate_listing_files(output_path: Path, subgraph_details: SerializedSegmentedNetwork, listing_files: dict, bundler: Bundler):
        subgraphs_count = len(subgraph_details.subgraphs)
        model_info_filename = os.path.join(os.path.dirname(output_path), 'model_info.txt')
        out_listing_files = [ model_info_filename ]
        with open(model_info_filename, 'w') as listing_file:
            listing_file.write(f'Model converted with SyNAP toolkit {pysynap.version}\n')

            if not subgraph_details.subgraph_metas:
                delegate_name = next(iter(subgraph_details.target_delegates.values())).type.value
                listing_file.write(f'\ndelegate: {delegate_name}')
            else:
                # Generate overall input/output information for the whole model
                listing_file.write(f'\ninputs:\n')
                for i, tensor in enumerate(subgraph_details.inputs):
                    tensor = get_tensor(subgraph_details, tensor)
                    listing_file.write(f'{tensor_to_str(tensor)}\n')
                listing_file.write(f'\noutputs:\n')
                for i, tensor in enumerate(subgraph_details.outputs):
                    tensor = get_tensor(subgraph_details, tensor)
                    listing_file.write(f'{tensor_to_str(tensor)}\n')
            listing_file.write(f'\nsubgraphs: {subgraphs_count}\n')

            # Append full input/output info for each subgraph
            listing_file.write('\n\n')
            for sg in bundler.subgraphs:
                listing_file.write('=' * 100 + '\n')
                listing_file.write(f'Detailed input/output information for {sg.name}:\n\n')
                with open(sg.meta_file) as json_meta_file:
                    # Read json from json_meta_file
                    # listing_file.write(json_meta_file.read())
                    metadata = json.load(json_meta_file)
                    metadata['inputs'] = [tensor_to_info(t) for t in metadata['Inputs'].values()]
                    metadata['outputs'] = [tensor_to_info(t) for t in metadata['Outputs'].values()]
                    del metadata['Inputs']
                    del metadata['Outputs']
                    if 'Recurrent_connections' in metadata:
                        del metadata['Recurrent_connections']
                    yaml = YAML()
                    yaml.indent(mapping=4, sequence=4, offset=2)
                    yaml.default_flow_style = None
                    yaml.dump(metadata, listing_file)
                    listing_file.write('\n')
                listing_file.write('\n')

        # Copy additional listing files from converters, appending files with the same name together
        # Fox example this will append the VSI NPU listing to model_info.txt
        for subgraph_name, subgraph_listings in listing_files.items():
            for listing_file in subgraph_listings:
                out_listing_file = os.path.join(os.path.dirname(output_path), os.path.basename(listing_file))
                if not out_listing_file in out_listing_files:
                    out_listing_files.append(out_listing_file)
                    with open(out_listing_file, 'w') as olf:
                        olf.write(f'Model converted with SyNAP toolkit {pysynap.version}\n')                    
                # Append this listing to the existing one with the same name
                with open(out_listing_file, 'a+') as olf:
                    olf.write('=' * 100 + '\n')
                    olf.write(f'Detailed graph information for {subgraph_name}:\n')
                    with open(listing_file) as ilf:
                        olf.write(ilf.read())


def pytorch_to_onnx(model_path: Path, meta: MetaInfo, work_dir: Path, verbose: bool):
    """
    Converts a PyTorch model to ONNX, simplifying it if possible.
    returns the path to the generated ONNX model
    """
    import torch.onnx
    import onnx
    import onnxsim

    logger.info(f"Loading torchscript model: {model_path}")
    try:
        model = torch.jit.load(model_path)
    except Exception as e:
        logger.error(f"Torch jit load error: {e}")
        raise ConversionError(f"{model_path} is not a valid torchscript model")
    
    logger.debug("Setting torch model to eval")
    model.eval()

    if not meta.input_names(False):
        # Automatically assign input names if not specified in metafile (reqired by acuity when shape specified)
        for i, name in enumerate(meta.inputs):
            meta.inputs[i].name = f'input{i}'

    logger.debug("Exporting torch model to onnx")
    onnx_model_path = work_dir / 'model.onnx'
    dummy_inputs = tuple([torch.randn(shape) for shape in meta.input_shapes()])
    torch.onnx.export(
        model, dummy_inputs, onnx_model_path,
        input_names = meta.input_names(False),
        output_names = meta.output_names(False),
        verbose=verbose,
        do_constant_folding = True
    )

    logger.debug("Loading back ONNX model to check it's ok")
    try:
        onnx_model = onnx.load(onnx_model_path)
        # onnx.checker.check_model(onnx_model)  # ultralytics/engine/exporter.py has this disabled
    except Exception as e:
        raise ConversionError(f"Failed to load back generated onnx model: {e}")

    # Try to simplify the model
    try:
        logger.info("Trying to simplify torchscript model..")
        simplified_onnx_model, check = onnxsim.simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        logger.info("Torchscript model simplified successfully")
        os.rename(onnx_model_path, work_dir / 'model_before_simplify.onnx')
        onnx.save(simplified_onnx_model, onnx_model_path)
    except Exception as e:
        logger.warning(f"Skipping ONNX model simplification due to simplifier error: {e}")

    return onnx_model_path
