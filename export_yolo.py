import argparse
import os
from pathlib import Path

from pysynap.tools.export.yolo import export_yolo_models
from pysynap.tools.convert import convert_multiple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def main() -> None:
    parser = argparse.ArgumentParser(
    prog=f"python export_yolo.py", description=__doc__
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="YOLOv8 or YOLOv9 model(s) (e.g.: yolov9c-seg, yolov8n, yolov8s-pose, ...)",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=["GENERIC", "DVF120", "VS640", "VS680", "SL1620", "SL1640", "SL1680"],
        help="Target SoC",
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        default=f"{os.getcwd()}/models/exported",
        metavar="DIR",
        help="Exported models directory (default: %(default)s)",
    )
    parser.add_argument(
        "--convert_dir",
        type=str,
        default=f"{os.getcwd()}/models/converted",
        metavar="DIR",
        help="Converted models directory (default: %(default)s)",
    )
    parser.add_argument(
        "--input_sizes",
        nargs="+",
        default=["640x352"],
        metavar="WIDTHxHEIGHT",
        help="Input image sizes. Each dimension must be a multiple of 32 (default: %(default)s)",
    )
    parser.add_argument(
        "--export_formats",
        nargs="+",
        default=["tflite"],
        metavar="FMT",
        choices=["tflite", "onnx", "pb", "pt"],
        help="Export model formats, select from [%(choices)s] (default: %(default)s)",
    )
    quant_grp = parser.add_argument_group("optional quantization parameters")
    quant_grp.add_argument(
        "--quant_types",
        nargs="+",
        metavar="TYPE",
        choices=["uint8", "int8", "int16", "float16", "mixed"],
        help="Quantization types to apply, select from [%(choices)s]",
    )
    quant_grp.add_argument(
        "--quant_dataset",
        type=str,
        metavar="FILE",
        nargs="+",
        help="Dataset(s) to be used for quantization",
    )
    parser.add_argument(
        "--profiling",
        action="store_true",
        default=False,
        help='Add "--profiling" during synap convert',
    )
    parser.add_argument(
        "--no_parallel",
        action="store_true",
        default=False,
        help="Disable parallel processing. Useful for resource constrained systems",
    )
    args = parser.parse_args()

    # export models to specified export formats.
    exported_paths: list[Path] = export_yolo_models(
        args.models,
        args.input_sizes,
        args.export_formats,
        args.quant_types,
        args.quant_dataset,
        args.export_dir,
        args.no_parallel
    )

    # convert exported models to synap
    convert_dir: Path = Path(args.convert_dir)
    convert_multiple(
        exported_paths,
        convert_dir,
        args.target,
        args.profiling,
        args.no_parallel
    )


if __name__ == "__main__":
    main()

