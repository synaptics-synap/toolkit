from typing import Any, Literal

import onnx
import tensorflow.lite as tf_lite
from google.protobuf.json_format import MessageToDict


__all__ = [
    "get_onnx_layer_info",
    "get_tflite_layer_info",
]


def get_onnx_layer_info(
    model_path: str,
    layer_name: Literal["input", "output"],
) -> list[dict[str, Any]]:
    if layer_name not in ("input", "output"):
        raise ValueError("Layer name must be one of ('input', 'output')")
    layer_info = []
    model = onnx.load(model_path)
    layer = model.graph.input if layer_name == "input" else model.graph.output
    for tensor in layer:
        tensor_info = {}
        tensor_info["name"] = tensor.name
        dim = tensor.type.tensor_type.shape.dim
        tensor_shape = [int(MessageToDict(d).get("dimValue")) for d in dim]
        tensor_info["shape"] = str(tensor_shape)
        layer_info.append(tensor_info)
    return layer_info


def get_tflite_layer_info(
    model_path: str,
    layer_name: Literal["input", "output"],
) -> list[dict[str, Any]]:
    if layer_name not in ("input", "output"):
        raise ValueError("Layer name must be one of ('input', 'output')")
    layer_info = []
    interpreter = tf_lite.Interpreter(model_path=model_path)
    layer = interpreter.get_input_details() if layer_name == "input" else interpreter.get_output_details()
    for tensor in layer:
        tensor_info = {}
        tensor_info["name"] = str(tensor["name"])
        tensor_shape = tensor["shape"].tolist()
        tensor_info["shape"] = str(tensor_shape)
        layer_info.append(tensor_info)
    return layer_info


if __name__ == "__main__":
    pass
