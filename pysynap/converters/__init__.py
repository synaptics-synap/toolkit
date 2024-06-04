# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from pysynap.graph_segmenter import SerializedSubgraph
from pysynap.meta import MetaInfo, DelegateInfo, NetworkFormat, Delegate


class TargetSOC(Enum):
    """
    Name of the target SOC
    """
    GENERIC = 'GENERIC'
    DVF120 = 'DVF120'
    VS640 = 'VS640'
    VS680 = 'VS680'
    SL1620 = 'SL1620'
    SL1640 = 'SL1640'
    SL1680 = 'SL1680'


@dataclass
class ConversionOptions:
    verbose: bool = False
    silent: bool = False
    debug: bool = False
    profiling: bool = False
    cpu_profiling: bool = False
    vssdk_dir: Path | None = None
    tools_dir: Path | None = None
    cache_dir: Path | None = None
    target: TargetSOC | None = None


@dataclass
class ConvertedModel:
    model_file: Path
    meta_file: Path
    listing_files: list[Path] | None = None
    input_count: int | None =  None
    output_count: int | None = None


def convert(subgraph: SerializedSubgraph, subgraph_meta: MetaInfo, delegate: DelegateInfo | None,
            conversion_options: ConversionOptions, work_dir: Path) -> ConvertedModel:
    # here we import the converters at the last possible time to prevent loops and avoid loading unnecessary modules
    if delegate.type == Delegate.NPU:
        from pysynap.converters.acuity_converter import convert
        return convert(subgraph, subgraph_meta, delegate, conversion_options, work_dir)
    elif delegate.type in [Delegate.CPU, Delegate.GPU]:
        if subgraph_meta.network_format == NetworkFormat.TFLITE:
            from pysynap.converters.tflite_converter import convert
            return convert(subgraph, subgraph_meta, delegate, conversion_options, work_dir)
        elif subgraph_meta.network_format == NetworkFormat.ONNX:
            from pysynap.converters.onnx_converter import convert
            return convert(subgraph, subgraph_meta, delegate, conversion_options, work_dir)
    else:
        raise Exception(f"Delegate {delegate} not supported")
