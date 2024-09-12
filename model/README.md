# SyNAP Model Export
This toolkit provides a convenient way of exporting models to the SyNAP format.
The following models are currently supported:

| Model | Export Script |
|-------|---------------|
| [YOLOv8](https://docs.ultralytics.com/models/yolov8/) | [export/yolo.py](/model/export/yolo.py) |
| [YOLOv9](https://docs.ultralytics.com/models/yolov9/) | [export/yolo.py](/model/export/yolo.py) |

## YOLO Export Quickstart
This tutorial will go through exporting the YOLOv8 segmentation (small) model into two input sizes (640x352, 224x224) and two quantization types (uint8, float16). The quantization dataset used is [`coco8-seg`](https://docs.ultralytics.com/datasets/segment/coco8-seg/) and is assumed to be located at `"/home/$USER/synaptics-synap/toolkit/datasets/coco8-seg/*.jpg"`. The board used is the Astra SL1680 with firmware v1.1.0.

### 1. Export YOLOv8 to TFLite models
```sh
python -m model.export.yolo \
    --models yolov8s-seg \
    --input_sizes 640x352 224x224 \
    --quant_types uint8 float16 \
    --quant_datasets "/home/$USER/synaptics-synap/toolkit/datasets/coco8-seg/*.jpg"
```
*See the [documentation on model export](/model/docs/export.md) for more details on* `model.export.yolo`

The above command will generate 4 TFLite models and their corresponding YAML conversion metafiles to the export directory, which by default is `models/exported`.
```
$ ls -1 models/exported
yolov8s-seg_224x224_tflite_float16.tflite
yolov8s-seg_224x224_tflite_float16.yaml
yolov8s-seg_224x224_tflite_uint8.tflite
yolov8s-seg_224x224_tflite_uint8.yaml
yolov8s-seg_640x352_tflite_float16.tflite
yolov8s-seg_640x352_tflite_float16.yaml
yolov8s-seg_640x352_tflite_uint8.tflite
yolov8s-seg_640x352_tflite_uint8.yaml
```
Now would be a good time to modify the conversion metafiles if needed. We'll be skipping that step for this tutorial and proceed directly to SyNAP conversion.

### 2. Converting TFLite models to SyNAP
```sh
python -m model.convert \
    --all \
    --target SL1680
```
*See the [documentation on model conversion](/model/docs/convert.md) for more details on* `model.convert`

The above command will convert all of the models from `models/exported` to the SyNAP format and store them in the converted models directory, which by default is `models/converted`.
```
$ tree models/converted -I *cache*
models/converted
├── yolov8s-seg_224x224_tflite_float16
│   ├── model.synap
│   └── model_info.txt
├── yolov8s-seg_224x224_tflite_uint8
│   ├── model.synap
│   ├── model_info.txt
│   └── quantization_info.yaml
├── yolov8s-seg_640x352_tflite_float16
│   ├── model.synap
│   └── model_info.txt
└── yolov8s-seg_640x352_tflite_uint8
    ├── model.synap
    ├── model_info.txt
    └── quantization_info.yaml
```
This step is identical to running `synap convert` manually on every model in `models/exported`.

### 3. Copying SyNAP models to board
```sh
python -m model.copy --all
```
Or via SSH:
```sh
python -m model.copy \
    --all \
    --board_ip <IP address>
```
*See the [documentation on copying models](/model/docs/copy.md) for more details on* `model.copy`

The above command will copy all SyNAP models from `models/converted` to the board at `root@<IP address>:/home/root/models` using SSH. The destination directory will be created if it doesn't exist.

### 4. [Optional] Profile copied models with `synap_cli`
```sh
python -m model.copy --all
```
Or via SSH:
```sh
python -m model.profile \
    --all \
    --board_ip <IP address>
```
*See the [documentation on profiling models](/model/docs/profile.md) for more details on* `model.profile`

The above command will profile all models with `synap_cli` from `root@<IP address>:/home/root/models` over SSH. The results will be displayed on the console:
```
yolov8s-seg_224x224_tflite_float16.synap: Inference timings (ms):  load: 186.23  init: 62.64  min: 892.30  median: 894.23  max: 899.63  stddev: 1.92  mean: 894.53
yolov8s-seg_224x224_tflite_uint8.synap: Inference timings (ms):  load: 72.76  init: 21.18  min: 7.98  median: 8.13  max: 14.90  stddev: 2.03  mean: 8.82
yolov8s-seg_640x352_tflite_float16.synap: Inference timings (ms):  load: 186.84  init: 64.22  min: 3972.01  median: 3973.22  max: 3982.74  stddev: 3.02  mean: 3974.20
yolov8s-seg_640x352_tflite_uint8.synap: Inference timings (ms):  load: 76.30  init: 22.54  min: 33.27  median: 33.41  max: 41.27  stddev: 2.37  mean: 34.15
```