# Profiling SyNAP Models on Board

| Implementation |
|----------------|
| [tools/profile.py](/tools/profile.py) |

This is a guide on profiling SyNAP models on a Astra board via SSH. ADB support is planned for the future.
The profiling results are from running `synap_cli` with `random` input and 10 inferences.

## Profile options
To profile models, run `python -m tools.profile`. The following options are available:
- `--serial`: The device serial for ADB. By default ADB will use the first connected device it finds, use this to specify a different device.
- `--board_ip`: The IP address of the board to profile models on. If provided, profiling will be done over SSH instead of ADB.
- `--models_dir`: Directory containing the models to profile. Default is `/home/root/models`.
- `--all | --models`: Select which models to copy from `--copy_dir`. Only one of these options may be specified at a time.
  - `--all`: Profile all models.
  - `--models NAME [NAME, ...]`: Profile all models corresponding to `NAME`s, which can be model filenames or a singular glob pattern.

A summarized version of this information is available via `python -m tools.convert --help`.

> [!NOTE]
> This tool is intended to be used in conjunction with [`tools.convert`](/tools/docs/copy.md). As such, model selection via `--models` is somewhat dependent on the format of the converted model filenames produced by the convert script.

## Profile examples
1. Profile all models:
```sh
python -m tools.profile \
    --all
```
2. Profile only models with 224x224 input size, over SSH:
```sh
python -m tools.profile \
    --models *224x224* \
    --board_ip <IP address>
```
3. Profile models from a different directory:
```sh
python -m tools.profile \
    --all \
    --models_dir /tmp
```

The script should ouput something as below:
```
Logged in: root@10.3.10.78
yolov8s_640x640_onnx.synap: Inference timings (ms):  load: 198.66  init: 61.26  min: 4889.62  median: 4895.05  max: 4904.47  stddev: 4.61  mean: 4895.80
yolov8s_640x640_onnx_uint8.synap: Inference timings (ms):  load: 77.64  init: 23.77  min: 54.29  median: 54.35  max: 62.54  stddev: 2.45  mean: 55.18
yolov8s_640x640_tflite.synap: Inference timings (ms):  load: 186.82  init: 57.09  min: 4884.80  median: 4886.66  max: 4893.94  stddev: 2.37  mean: 4887.30
yolov8s_640x640_tflite_uint8.synap: Inference timings (ms):  load: 77.98  init: 23.71  min: 53.06  median: 53.17  max: 61.08  stddev: 2.38  mean: 53.95
```