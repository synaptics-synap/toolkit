import onnx
import tensorflow.lite as tf_lite
from google.protobuf.json_format import MessageToDict


__all__ = [
    "get_onnx_inp_info",
    "get_tflite_inp_info",
    "get_onnx_num_outputs",
    "get_tflite_num_outputs",
]


def get_onnx_inp_info(model_path: str) -> list[dict[str, str | int]]:
    inputs_info = []
    model = onnx.load(model_path)
    for _input in model.graph.input:
        inp_info = {}
        inp_info["name"] = _input.name
        dim = _input.type.tensor_type.shape.dim
        inp_shape = [int(MessageToDict(d).get("dimValue")) for d in dim]
        inp_info["shape"] = str(inp_shape)
        inputs_info.append(inp_info)
    return inputs_info


def get_tflite_inp_info(model_path: str) -> list[dict[str, str | int]]:
    inputs_info = []
    interpreter = tf_lite.Interpreter(model_path=model_path)
    for _input in interpreter.get_input_details():
        inp_info = {}
        inp_info["name"] = str(_input["name"])
        inp_shape = list(_input["shape"])
        inp_info["shape"] = str(inp_shape)
        inputs_info.append(inp_info)
    return inputs_info


def get_onnx_num_outputs(model_path: str) -> int:
    model = onnx.load(model_path)
    return len(model.graph.output)


def get_tflite_num_outputs(model_path: str) -> int:
    interpreter = tf_lite.Interpreter(model_path)
    return len(interpreter.get_output_details())


if __name__ == "__main__":
    pass
