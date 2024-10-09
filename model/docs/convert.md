# Converting Models with SyNAP

| Implementation |
|----------------|
| [model/convert.py](/model/convert.py) |

This is a guide on converting suitable models to the SyNAP model format.
It is recommended to review the [documentation on the SyNAP model converter](https://synaptics-synap.github.io/doc/manual/working_with_models.html#model-conversion) first, as [convert.py](/model/convert.py) internally uses the SyNAP model converter.

## Conversion options
Model conversion can be done with `python -m model.convert`. The following options are available:
- `--export_dir`: Directory containing the models to convert. Default is `models/exported`.
- `--convert_dir`: Directory to store the converted models. Default is `models/converted`.
- `--all | --latest | --models`: Select which models to convert from `--export_dir`. Only one of these options may be specified at a time.
  - `--all`: Convert all models.
  - `--latest`: Convert the most recently exported model.
  - `--models NAME [NAME, ...]`: Convert all models corresponding to `NAME`s, which can be model filenames or a singular glob pattern.
- `--target`: The target SoC, corresponds to `--target` from `synap convert`.
- `--profiling`: Enable model profiling during conversion, corresponds to `--profiling` from `synap convert`.
- `--export_formats`: Only models of this format will be considered for conversion. Default is `["onnx", "tflite"]`.
- `--no_parallel`: Disables converting multiple models in parallel. Can be useful if the host machine is resource constrained.

A summarized version of this information is available via `python -m model.convert --help`.

> [!NOTE]
> This tool is intended to be used in conjunction with [model export](/model/docs/export.md). As such, model selection via `--models` is somewhat dependent on the format of the exported model filenames produced by the export script.

## Conversion examples for SL1680
1. Convert all models:
```
python -m model.convert \
    --all \
    --target SL1680
```
2. Convert only models with 224x224 input size:
```
python -m model.convert \
    --models *224x224* \
    --target SL1680
```
3. Convert specific models by specifying model filenames (relative to `--export_dir`):
```
python -m model.convert \
    --models yolov8m_224x224_onnx_uint8.onnx yolov8n-pose_224x224_tflite_float16.tflite \
    --target SL1680
```
4. Convert a model from a different directory:
```
python -m model.convert \
    --latest \
    --target SL1680 \
    --export_dir "my_models_dir"
```