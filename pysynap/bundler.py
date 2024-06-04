# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

import encodings
from dataclasses import dataclass
from pathlib import Path
import json
import zipfile

from pysynap.graph_segmenter import SubgraphName, SubgraphTensor, GlobalTensorId
from pysynap.meta import Delegate


@dataclass
class Subgraph:
    """
    Information of a subgraph that needs to be bundled
    """
    name: SubgraphName
    input_buffers: list[GlobalTensorId]
    output_buffers: list[GlobalTensorId]
    model_file: Path
    meta_file: Path
    delegate: Delegate


class Bundler:
    """
    Generates a .synap bundle containing a set of subgraphs and their metadata
    """

    VERSION = '0.0.1'

    def __init__(self):
        self.subgraphs: list[Subgraph] = []
        self.tensors: dict[GlobalTensorId, SubgraphTensor] = {}
        self.inputs: list[GlobalTensorId] = []
        self.outputs: list[GlobalTensorId] = []

    def _add_tensor(self, tensors: list[SubgraphTensor]):
        for tensor in tensors:
            if tensor.global_tensor_id in self.tensors:
                other_tensor = self.tensors[tensor.global_tensor_id]
                if tensor.data_type != other_tensor.data_type or tensor.shape != other_tensor.shape:
                    raise Exception("Meta info changed")
            else:
                self.tensors[tensor.global_tensor_id] = tensor


    # use ZipInfo struct to write the file.
    # This is needed to ensure that the generated zip file content is deterministic and doesn't
    # depend on the creation date of the files
    @staticmethod
    def zip_write(zip_fp, disk_file_name:str, zip_file_name:str):
        with open(disk_file_name, 'rb') as f:
            zip_fp.writestr(zipfile.ZipInfo(filename=zip_file_name), f.read())


    def add_subgraph(self, subgraph_name: SubgraphName, input_tensors: list[SubgraphTensor],
                     output_tensors: list[SubgraphTensor], model_filename: Path, meta_filename: Path,
                     delegate: Delegate):
        """
        Add a subgraph that needs to be bundled
        """

        input_buffer_ids = [buf.global_tensor_id for buf in input_tensors]
        output_buffer_ids = [buf.global_tensor_id for buf in output_tensors]
        subgraph = Subgraph(name=subgraph_name,
                            input_buffers=input_buffer_ids,
                            output_buffers=output_buffer_ids,
                            model_file=model_filename,
                            meta_file=meta_filename,
                            delegate=delegate)

        self._add_tensor(input_tensors)
        self._add_tensor(output_tensors)
        self.subgraphs.append(subgraph)

    def create_bundle_data(self):
        """
        Create the bundle data that will be serialized to bundle.json.

        This outputs the interim format that the target code can parse at the moment.

        TODO: converge to final format and adapt this code
        """

        # find all the tensors that are outputted by the subgraphs and construct a
        # dictionary with the subgraph index and output index
        tensor_id_to_subgraph_out: dict[GlobalTensorId, dict] = {}
        for subgraph_idx, subgraph in enumerate(self.subgraphs):
            for out_idx, tensor_id in enumerate(subgraph.output_buffers):
                tensor_id_to_subgraph_out[tensor_id] = {'subgraph': subgraph_idx, 'out': out_idx}

        # append also all the network inputs to the mapping
        for input_idx, tensor_id in enumerate(self.inputs):
            spec = {'subgraph': -1, 'out': input_idx}
            tensor_id_to_subgraph_out[tensor_id] = spec

        # create the bundle.json data
        subgraphs = []

        for idx, subgraph in enumerate(self.subgraphs):

            subgraph_inputs: list[dict] = []

            for global_tensor_id in subgraph.input_buffers:
                subgraph_inputs.append(tensor_id_to_subgraph_out[global_tensor_id])

            # list all buffers that are computed by other subgraphs
            subgraph_inputs = [tensor_id_to_subgraph_out[x] for x in subgraph.input_buffers
                               if x in tensor_id_to_subgraph_out]

            subgraphs.append({'inputs': subgraph_inputs,
                              'model': str(idx) + '/' + subgraph.model_file.name,
                              'meta': str(idx) + '/' + subgraph.meta_file.name
                              })

        return {'graph': subgraphs, 'outputs': [tensor_id_to_subgraph_out[x] for x in self.outputs]}

    def create_bundle(self, out_path: Path):
        """
        Creates a bundle with the subgraphs that were added so far at the specified location
        """

        with zipfile.ZipFile(out_path, 'w') as zip_fp:

            # write out the bundle file
            with zip_fp.open('bundle.json', 'w') as fp:
                json.dump(self.create_bundle_data(), encodings.utf_8.StreamWriter(fp), indent=True)

            for idx, subgraph in enumerate(self.subgraphs):
                self.zip_write(zip_fp, subgraph.model_file, f'{idx}/{subgraph.model_file.name}')
                self.zip_write(zip_fp, subgraph.meta_file, f'{idx}/{subgraph.meta_file.name}')

        return out_path
