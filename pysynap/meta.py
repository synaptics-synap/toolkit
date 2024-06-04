# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

import re
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import NewType

import ruamel.yaml

from .exceptions import *
from enum import Enum

from .network import OperationPattern, TensorName, DataType, TensorShape

DataFormat = NewType('DataFormat', str)
"""
Format of the data of an input or output. This is used by pre-processing code and post-processing code to understand
what kind of transformation must be applied. This not just an identifier but is astring that may contain information
about the details of the format.
"""


class NetworkFormat(Enum):
    TFLITE = 'tflite'
    TENSORFLOW = 'pb'
    ONNX = 'onnx'
    CAFFE = 'caffe'
    TORCH = 'torch'
    NCNN = 'ncnn'

UNKNOWN_DATA_FORMAT = DataFormat('')


class QuantizerAlgorithm(Enum):
    STANDARD = 'standard'
    MOVING_AVERAGE = 'moving_average'
    KL_DIVERGENCE = 'kl_divergence'
    AUTO = 'auto'


class DataLayout(Enum):
    DEFAULT = 'default'
    NCHW = 'nchw'
    NHWC = 'nhwc'


PreprocessingJson = NewType('PreprocessingJson', dict)
InputSecurityPolicy = NewType('InputSecurityPolicy', str)
OutputSecurityPolicy = NewType('OutputSecurityPolicy', str)


DEFAULT_INPUT_SECURITY = InputSecurityPolicy('any')


@dataclass
class PreprocInfo:
    """
    Information for the preprocessing that the runtime must apply to the given input data.

    This is copied to the meta JSON used by the delegate that uses the input.
    """
    preproc_type: str
    size: list[int]
    crop_rect: list[int]


@dataclass
class InputInfo:
    """
    Represented the information about the input of the network that needs to be produced during conversion.

    This is used by the conversion tool used to generate the target network.
    """

    name: TensorName | None
    """
    Name of the tensor in the original network that is the source of this output, when it
    is None the output used is the output at the same position in the original network.

    It is guaranteed that a tensor with a given name is present only once in the inputs.
    """

    means: float | list[float] | None
    scale: float | int | None
    data_format: DataFormat
    security: InputSecurityPolicy
    preproc: PreprocInfo | None
    tensor_h: None
    tensor_w: None
    shape: TensorShape | None

    def get_means(self, channels: int):
        if isinstance(self.means, list):
            if len(self.means) != channels:
                raise ConversionError(f"bad item count in 'means': {self.means}. "
                                      "Input {self.name if self.name else ''} has {channels} channels")
            return self.means
        return [self.means] * channels


DEFAULT_OUTPUT_SECURITY = OutputSecurityPolicy('secure-if-input-secure')


@dataclass
class OutputInfo:
    """
    Describes an output of the network that need to be generated.

    This is used by the conversion tool to generate the target network.
    """

    name: TensorName | None
    """
    Name of the tensor in the original network that is the source of this output, when it
    is None the output used is the output at the same position in the original network.

    It is guaranteed that a tensor with a given name is present only once in the outputs.
    """
    dequantize: bool
    data_format: DataFormat
    security: OutputSecurityPolicy
    shape: TensorShape | None


@dataclass
class QuantizationInfo:
    """
    Information used to quantize parts of the network
    """

    dataset: list[str]
    scheme: str
    data_type: DataFormat | dict[str, DataFormat]
    mode: str
    algorithm: QuantizerAlgorithm
    options: dict


@dataclass
class SecurityInfo:
    """
    Information about the files required to encrypt a model for a target device
    """

    encryption_key: str
    signature_key: str
    model_certificate: str
    vendor_certificate: str


class Delegate(Enum):
    """
    Name of the delegate that can run subgraphs
    """

    DEFAULT = 'default'
    NPU = 'npu'
    GPU = 'gpu'
    CPU = 'cpu'


@dataclass
class DelegateInfo:
    """
    Delegate type and attributes.
    """
    type: Delegate
    options: str


@dataclass
class MetaInfo:
    """
    Metadata used to convert the input network
    """

    network_format: NetworkFormat = NetworkFormat.TFLITE
    """
    Format of the input network
    """

    layout: DataLayout = DataLayout.DEFAULT
    """
    Layout of input and output tensors
    """

    input_format: DataFormat = UNKNOWN_DATA_FORMAT
    """
    Default input tensor format in case it is not specified in the InputInfo
    """

    output_format: DataFormat = UNKNOWN_DATA_FORMAT
    """
    Default output tensor format in case it is not specified in the OutputInfo
    """

    optimize: bool = True
    dequantize_outputs: bool = False

    inputs: list[InputInfo] = field(default_factory=list)
    """
    Input tensors of the network, these may be different from the input tensors of the underlying model file.

    Each input info corresponds to one input of the generated network, the source of this input is either
    the tensor with the same name in the original network or the output at the same position in the original network.

    When the input is an empty list all the inputs of the model being converted should be used.
    """

    outputs: list[OutputInfo] = field(default_factory=list)
    """
    Output tensors of the network, these may be different from the output tensors of the underlying model file

    Each output info corresponds to one output of the generated network, the source of this output is either
    the tensor with the same name in the original network or the output at the same position in the original network.

    When the outputs is an empty list all the inputs of the model being converted should be used.
    """

    quantization: QuantizationInfo | None = None
    """
    Quantization information for the network, None if the network shall not be quantized
    """

    security: SecurityInfo | None = None
    """
    Security information for the network, None if the network shall not be protected
    """

    delegates: dict[OperationPattern, DelegateInfo] = field(default_factory=dict)
    """
    Rules to map operators to delegates that must run them
    """

    @staticmethod
    def _names(desc, items, mandatory: bool = True) -> list | None:
        if mandatory:
            if not items:
                raise ConversionError(f"Missing required '{desc}s' field in meta file")
            if any(not i.name for i in items):
                raise ConversionError(f"Missing required {desc} 'name' field in meta file")
        if any(i.name for i in items) and any(not i.name for i in items):
            raise ConversionError(f"All {desc} names or none must be specified in meta file")
        names = [i.name for i in items] if any(i.name for i in items) else None
        return names

    def input_names(self, mandatory: bool = True) -> list | None:
        return self._names("input", self.inputs, mandatory)

    def output_names(self, mandatory: bool = True) -> list | None:
        return self._names("output", self.outputs, mandatory)

    def input_shapes(self, mandatory: bool = True) -> list | None:
        # Check all required info have been provided
        if mandatory:
            if not self.inputs:
                raise ConversionError(f"Missing required 'inputs' field in meta file")
            if any(i.shape is None for i in self.inputs):
                raise ConversionError(f"Missing required input 'shape' field in meta file")
        if any(i.shape for i in self.inputs) and any(not i.shape for i in self.inputs):
            raise ConversionError(f"All input shapes or none must be specified in meta file")
        if not any(i.shape for i in self.inputs):
            return None
        return [i.shape for i in self.inputs]


    @staticmethod
    def _names_str(desc, items, mandatory: bool = True) -> str | None:
        names = MetaInfo._names(desc, items, mandatory)
        return " ".join(names) if names else None

    def input_names_str(self, mandatory: bool = True) -> str | None:
        return self._names_str("input", self.inputs, mandatory)

    def output_names_str(self, mandatory: bool = True) -> str | None:
        return self._names_str("output", self.outputs, mandatory)

    def input_shapes_str(self, mandatory: bool = True) -> str | None:
        shapes = self.input_shapes(mandatory)
        # Remove 1st dimension (N) from shapes and concatenate all inputs with # separator
        # e.g. for 2 inputs [1, 224, 224, 3] -> "224,224,3#224,224,3"
        return "#".join([','.join([str(s) for s in shape[1:]]) for shape in shapes]) if shapes else None

    def check_inputs_with_shape_have_name(self):
        for i in self.inputs:
            if i.shape is not None and not i.name:
                raise ConversionError("Input name is mandatory when shape specified: " + str(i.shape))


def _check_fields(data, valid_fields: list[str], mandatory_fields: list[str] | None = None):
    """
    Checks that data is a dictionary and that contains all mandatory fields and any other field is listed in the
    valid_fields list
    """

    if mandatory_fields is None:
        mandatory_fields = []

    if not isinstance(data, dict):
        raise ConversionError(f"Invalid field in yaml file: " + str(data) + " is not a dictionary")

    fields = set(data.keys())
    diff = fields - set(valid_fields + mandatory_fields)

    if diff:
        raise ConversionError(f"Invalid field(s) in yaml file: " + ", ".join(list(diff)))

    if mandatory_fields:
        diff = set(mandatory_fields) - fields
        if diff:
            raise ConversionError(f"Missing mandatory field(s) in yaml file: " + ", ".join(list(diff)))


def _check_file_exists(file_name: Path, desc: str):
    """
    Check that file exists
    """

    if not file_name.is_file():
        raise ConversionError(f"{desc} not found: {file_name}")


def _preprocess_file(s: str | None, base_path: Path):
    """
    Replace {FILE:filename} in string with the content of file "filename"
    filename is absolute or relative to the directory containing the metafile
    """

    if not s:
        return ''

    while True:
        match = re.search(r'\${FILE:([^}]*)}', s)
        if not match:
            break
        filename = Path(match.group(1))
        filepath = base_path / filename
        _check_file_exists(filepath, f"File in string")
        with open(filepath, "r") as f:
            s = s.replace('${FILE:' + str(filename) + '}', f.read())

    return s


def _preprocess_env_var(s: str | None):
    """
    Replace {ENV:varname} in a string with the content the corresponding environment variable
    If the environment variable is not defined it is assumed the empty string
    """
    if not s:
        return ''
    while True:
        match = re.search(r'\${ENV:([^}]*)}', s)
        if not match:
            break
        var_name = match.group(1)
        var_content = os.environ.get(var_name, '')
        s = s.replace('${ENV:' + var_name + '}', var_content)
    return s


def _load_yaml(file_name: Path, desc: str = 'yaml file'):
    """
    Load, preprocess and parse a yaml file
    """

    _check_file_exists(file_name, desc)
    try:
        with open(file_name, "r") as fp:
            yaml_content = ruamel.yaml.YAML().load(_preprocess_env_var(fp.read()))
    except Exception as e:
        raise ConversionError(f"Error parsing {desc}: {file_name}") from e

    if yaml_content is None:
        yaml_content = {}
    elif not isinstance(yaml_content, dict):
        raise ConversionError(f"Invalid format for {desc}: {file_name}")

    return yaml_content


def _parse_preproc(data) -> 'PreprocInfo':
    _check_fields(data, ['crop', 'size'], ['type'])
    crop_rect = data.get('crop')
    if isinstance(crop_rect, bool):
        # Use dummy crop rect, the values are not important as long as w is > 0 and
        # w,h are different from those of the input tensor.
        # Using a big value ensures that pre_process_nv12_scale_U8toU8 is selected which seems to
        # give better results than pre_process_nv12_scale_U8toU8_gq when downscaling more than 4 times.
        # Actual coordinates will have to be specified at runtime anyway.
        crop_rect = [0, 0, 30000, 30000] if crop_rect else None

    return PreprocInfo(
        preproc_type=data.get('type'),
        size=data.get('size'),
        crop_rect=crop_rect
    )


def _parse_input_info(data, base_path: Path) -> InputInfo:
    if data is None:
      data = {}

    _check_fields(data, ['name', 'shape', 'means', 'scale', 'format', 'security', 'custom', 'preprocess'])

    name = data.get('name')
    shape = data.get('shape')
    means = data.get('means')
    scale = data.get('scale')
    data_format = _preprocess_file(data.get('format', UNKNOWN_DATA_FORMAT), base_path)
    preproc = data.get('preprocess')
    security = data.get('security', DEFAULT_INPUT_SECURITY)

    if name is not None:
        if not isinstance(name, str):
            raise ConversionError("Input name '" + str(name) + "' must be a string when specified")

    if shape is not None:
        # The following test is not actually needed in caffe, so check separately
        # if not self.name:
        #    raise ConversionError("Input name is mandatory when shape specified: " + str(shape))
        if not isinstance(shape, list) or not all(isinstance(n, int) for n in shape):
            raise ConversionError("Shape for input '" + name + "' must be a list of integers: " + str(shape))
        if len(shape) < 2:
            raise ConversionError("Shape for input '" + name + "' doesn't include batch dimension: " + str(shape))

    if name and not shape:
        raise ConversionError("Input shape is mandatory when name specified: " + str(name))

    if means is not None:
        if not isinstance(means, (list, int, float)):
            raise ConversionError(f"invalid means provided for input {name if name else ''}: {means}")

    if scale is not None:
        if not isinstance(scale, (int, float)) or scale == 0:
            raise ConversionError(f"invalid scale provided for input {name if name else ''}: {scale}")

    if preproc is not None:
        preproc = _parse_preproc(preproc)

    return InputInfo(name=name,
                     means=means,
                     scale=scale,
                     data_format=DataFormat(data_format),
                     security=security,
                     preproc=preproc,
                     tensor_h=None,
                     tensor_w=None,
                     shape=shape)


def _parse_output_info(data, base_path: Path) -> OutputInfo:
    if data is None:
      data = {}

    _check_fields(data, ['name', 'shape', 'dequantize', 'format', 'security', 'custom'])

    return OutputInfo(
        name=data.get("name"),
        dequantize=data.get('dequantize', False),
        data_format=DataFormat(_preprocess_file(data.get('format', ''), base_path)),
        security=data.get('security', DEFAULT_OUTPUT_SECURITY),
        shape=data.get('shape')
    )


def _parse_quantization_info(data, base_dir: Path):
    _check_fields(data, ['dataset', 'scheme', 'data_type', 'mode', 'algorithm', 'options'])

    dataset = data.get('dataset')
    scheme = data.get('scheme')
    data_type = data.get('data_type', DataType.UINT8.value)
    mode = data.get('mode', 'standard')
    algorithm_name = data.get('algorithm', 'standard')
    options = data.get('options', {})

    if options is None:
        options = {}

    if not isinstance(options, dict):
        raise ConversionError(f"Invalid quantization options specified: {options}")

    valid_algorithms = [x.value for x in QuantizerAlgorithm]

    if algorithm_name not in valid_algorithms:
        raise ConversionError(f'Invalid quantization algorithm specified: {algorithm_name} not in {valid_algorithms}')

    dt = data_type

    try:
        if isinstance(data_type, str) and data_type != DataType.FLOAT16.value:
            data_type = DataType(dt)
        elif isinstance(data_type, dict):
            for spec, dt in data_type.items():
                data_type[spec] = DataType(dt)
            data_type = data_type
        else:
            raise ConversionError(f'invalid quantization data_type specified: {dt}')
    except ValueError:
        raise ConversionError(f'invalid quantization data_type specified: {dt}')

    if isinstance(dataset, str):
        dataset = [dataset]

    # Make dataset paths absolute if not already so
    if dataset:
        dataset = [str((Path(base_dir) / d).absolute()) if d != 'random' else d for d in dataset]

    return QuantizationInfo(dataset, scheme, data_type, mode, QuantizerAlgorithm(algorithm_name), options)


def _parse_security_info(data, base_path: Path) -> SecurityInfo | None:
    sec_fields = ['encryption_key', 'signature_key', 'model_certificate', 'vendor_certificate']
    secure = data.get('secure', True)

    if not secure:
        return None

    security_file = data.get('file')

    if security_file is not None:
        _check_fields(data, ['secure', 'file'])
        security_file_path = base_path / security_file
        security_base_path = security_file_path.parent
        security = _load_yaml(security_file_path, 'security file')
        _check_fields(security, [], sec_fields)
    else:
        security_base_path = base_path
        security = data
        _check_fields(data, ['secure'], sec_fields)

    encryption_key = security_base_path / security.get('encryption_key')
    signature_key = security_base_path / security.get('signature_key')
    model_certificate = security_base_path / security.get('model_certificate')
    vendor_certificate = security_base_path / security.get('vendor_certificate')

    _check_file_exists(encryption_key, 'encryption_key file')
    _check_file_exists(signature_key, 'signature_key file')
    _check_file_exists(model_certificate, 'model_certificate file')
    _check_file_exists(vendor_certificate, 'vendor_certificate file')

    return SecurityInfo(encryption_key=encryption_key, signature_key=signature_key,
                        model_certificate=model_certificate, vendor_certificate=vendor_certificate)


def _parse_delegate_info(info: str) -> DelegateInfo:
    info = info.split(' ', 1)
    valid_delegate_types = [d.value for d in Delegate]
    delegate_type = info[0]
    if not delegate_type:
        delegate_type = Delegate.DEFAULT.value
    if not delegate_type in valid_delegate_types:
        raise ConversionError(f"Invalid delegate type: {delegate_type} not in {valid_delegate_types}")
    return DelegateInfo(Delegate(delegate_type), info[1] if len(info) > 1 else '')


def load_metafile(meta_file: Path):
    meta_file = meta_file.absolute()
    base_path = meta_file.parent
    model_meta = _load_yaml(meta_file, 'model metafile')

    # Check fields in meta-file
    _check_fields(model_meta, [
        'optimize',
        'data_layout',
        'input_format',
        'output_format',
        'denormalize_outputs',
        'dequantize_outputs',
        'inputs',
        'outputs',
        'quantization',
        'security',
        'custom',
        'delegate'
    ])

    # Get input info from meta-file
    optimize = model_meta.get('optimize', True)
    layout_str = str(model_meta.get('data_layout', DataLayout.DEFAULT.value)).lower()
    input_format = str(model_meta.get('input_format', '')).lower()
    output_format = str(model_meta.get('output_format', '')).lower()
    dequantize_outputs = (model_meta.get('denormalize_outputs', False) or model_meta.get('dequantize_outputs', False))

    delegates = model_meta.get('delegate')
    if not delegates:
        delegates = Delegate.DEFAULT.value
    if isinstance(delegates, str):
        delegates  = {'*': delegates}
    delegates = {x: _parse_delegate_info(y) for x, y in delegates.items()}

    if layout_str:
        try:
            layout = DataLayout(layout_str)
        except ValueError:
            raise ConversionError('Invalid data_format specified: ' + layout_str)
    else:
        layout = None

    inputs = [_parse_input_info(data, base_path) for data in model_meta.get("inputs", [])]

    input_names = [i.name for i in inputs if i.name]
    if len(input_names) != len(set(input_names)):
        raise ConversionError(f"Duplicate input names in: {meta_file}")

    # Get output info from meta-file
    outputs = [_parse_output_info(data, base_path) for data in model_meta.get("outputs", [])]

    output_names = [o.name for o in outputs if o.name]
    if len(output_names) != len(set(output_names)):
        raise ConversionError(f"Duplicate output names in: {meta_file}")

    # Get quantization info from meta-file
    quantization = model_meta.get('quantization')

    if quantization is not None:
        q = _parse_quantization_info(quantization, base_path)
        if q.dataset is None:
            # Disable quantization if no dataset
            q = None
        else:
            if len(inputs) != len(q.dataset):
                raise ConversionError(f"Metafile has {len(inputs)} input(s) but {len(q.dataset)} quantization dataset")
    else:
        q = None

    # Get security info from meta-file or security file
    security = model_meta.get('security')
    if security is not None:
        security = _parse_security_info(security, base_path)

    return MetaInfo(layout=layout,  input_format=DataFormat(input_format),
                    output_format=DataFormat(output_format), optimize=optimize,
                    dequantize_outputs=dequantize_outputs, inputs=inputs,
                    outputs=outputs, quantization=q, security=security,
                    delegates=delegates)
