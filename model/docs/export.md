# Exporting Models

| Implemenation |
|---------------|
| [export/base.py](/model/export/base.py) |
| [export/yolo.py](/model/export/yolo.py) |

This is a guide on exporting models to various SyNAP compatible formats.
The base model export file is located at [base.py](/model/export/base.py) and contains classes and functions for creating custom model export scripts and conversion metafile generation.

> [!NOTE]
> Conversion metafile generation is currently only supported for TFLite (`.tflite`) and ONNX (`.onnx`) models.

## Exporting YOLO models
| Model | Export Script |
|-------|---------------|
| [YOLOv8](https://docs.ultralytics.com/models/yolov8/) | [export/yolo.py](/model/export/yolo.py) |
| [YOLOv9](https://docs.ultralytics.com/models/yolov9/) | [export/yolo.py](/model/export/yolo.py) |

The `export/yolo.py` script can be used to export YOLOv8 and YOLOv9 models to TFLite (`.tflite`), ONNX (`.onnx`), PyTorch (`.pt`) and Tensorflow (`.pb`) formats. The script can be run with `python -m model.export.yolo` and has the following options:
- `--models`*: The YOLO model(s) to export. Can be a YOLO model name (e.g. "yolov8s-seg") or a saved model weights (`.pt`) file.
- `--export_dir`: The directory to save exported models and their converstion metafiles in. The default is `"models/exported"`.
- `--input_sizes`*: Model input size(s) to export in, specified as `widthxheight`. Both width and height must be multiples of 32. The default is `640x352`.
- `--export_formats`*: Which formats to export. The default is `.tflite`.
- `--no_parallel`: Disables exporting multiple models in parrallel. Can be useful if the host machine is resource constrained.
- `--quant_types`*: Quantization scheme(s) to apply, choose from ["uint8", "int8", "int16", "float16", "mixed"].
  - The default is no quantization.
  - This parameter doesn't affect the actual model export but rather the contents of the conversion metafile.
- `--quant_datasets`: The dataset(s) to use for quantization.
  - This is required if `--quant_type` is specified and contains schemes other that "float16".
  - Can be individual files or glob patterns like `"dataset/*.jpg"`.
  - As with `--quant_types` this parameter only affects the contents of the conversion metafile.

> [!TIP]
> The script will do a cartesian product to compute all possible combinations for options marked with a *, and export a model for each combination.

Summarzied info on the input options is available via `python -m model.export.yolo --help`.

### YOLO export examples
1. Exporting YOLOv9 (compact) to multiple export formats (2 exported models):
```
python -m model.export.yolo \
    --models yolov9c \
    --export_formats onnx tflite
```
2. Exporting YOLOv8 segmentation (nano) to multiple input sizes (2 exported models):
```
python -m model.export.yolo \
    --models yolov8n-seg \
    --input_sizes 640x352 224x224
```
3. Exporting a model from saved weights to multiple quantization schemes (2 exported models):
```
python -m model.export.yolo \
    --models "my_fine_tuned_model.pt" \
    --quant_types uint8 float16 \
    --quant_datasets "datasets/coco8-seg/*.jpg"
```
4. Exporting multiple YOLO models to multiple export formats, input sizes, and quantization schemes (24 exported models):
```
python -m model.export.yolo \
    --models yolov8m yolov8s-seg yolov8n-pose \
    --input_sizes 640x352 224x224 \
    --export_formats onnx tflite \
    --quant_types uint8 float16 \
    --quant_datasets "datasets/coco8-seg/*.jpg"
```

## Creating Exporters
The `ModelExporter` class from [export/base.py](/model/export/base.py) can be extended to create custom exporters for different models.

[WIP]
