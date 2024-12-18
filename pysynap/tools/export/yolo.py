"""Export YOLOv8 and YOLOv9 models to various formats"""

import argparse
import os
from itertools import product
from logging import ERROR
from pathlib import Path
from shutil import rmtree
from typing import Any

from ultralytics import YOLO
from ultralytics.utils import LOGGER

from ..export.base import ModelExporter, ModelExportInfo, export_and_save_models
from ..utils.model_info import *

__all__ = [
    "YOLOModelExporter",
    "export_yolo_models"
]

# suppress Ultralytics logging
LOGGER.setLevel(ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class YOLOModelExporter(ModelExporter):

    def get_out_fmt_name(self) -> str:
        parts = self.model_info.model_name.split("-")
        out_fmt_name = parts[0][:-1]
        if len(parts) > 1 and parts[1] == "seg":
            return out_fmt_name + "seg"
        return out_fmt_name

    def export_model(self) -> str:
        if self.model_info.saved_weights:
            model = YOLO(self.model_info.saved_weights)
        else:
            model = YOLO(self.model_info.model_name + ".pt")
        export_args: dict[str, Any] = {}
        if self.model_info.export_format == "onnx":
            export_args.update({"simplify": True, "opset": 12})
        export_path: str = model.export(
            format=self.model_info.export_format,
            imgsz=(self.model_info.inp_height, self.model_info.inp_width),
            **export_args,
        )
        return export_path
    
    def generate_input_metadata(self, model_path: str) -> list[dict] | None:
        inputs_info: list[dict] | None = super().generate_input_metadata(model_path)
        if inputs_info is not None:
            for input_info in inputs_info:
                input_info["format"] = "rgb keep_proportions=1"
                input_info["scale"] = 255
        return inputs_info
    
    def generate_output_metadata(self, model_path: str) -> list[dict] | None:
        outputs_info: list[dict] | None = super().generate_output_metadata(model_path)
        if outputs_info is not None:
            bb_norm: int = 1 if self.model_info.export_format != "onnx" else 0
            for output_info in outputs_info:
                output_info["format"] = f"{self.get_out_fmt_name()} w_scale={self.model_info.inp_width} h_scale={self.model_info.inp_height} bb_normalized={bb_norm}"
                output_info["dequantize"] = True
        return outputs_info
    
    def cleanup_export_files(self) -> None:
        saved_weights = self.model_info.saved_weights.name if self.model_info.saved_weights else None
        work_dir: Path = Path(".")
        if (runs_dir := work_dir / "runs").exists():
            rmtree(runs_dir)
        for f in work_dir.iterdir():
            if f.is_dir():
                rmtree(f)
            if f.is_file():
                if f.suffix.lstrip(".") == "npy":
                    pass
                elif f.suffix.lstrip(".") == "pt" and f.name != saved_weights:
                    pass
                else:
                    f.unlink()


def export_yolo_models(
    models: list[str],
    input_sizes: list[str],
    export_formats: list[str],
    quant_types: list[str],
    quant_dataset: str,
    export_dir: str,
    no_parallel: bool
) -> list[Path]:
    inp_sizes: list[tuple[int, int]] = [
        tuple(int(dim) for dim in size.split("x")) for size in input_sizes
    ]

    exporters: list[YOLOModelExporter] = []
    for info in product(models, inp_sizes, export_formats):
        model, (inp_width, inp_height), export_format = info
        saved_weights = None
        if Path(model).suffix == ".pt":
            try:
                saved_weights = Path(model).resolve(strict=True)
                try:
                    model = YOLO(model)
                    model = Path(model.model.yaml["yaml_file"]).stem
                except (AttributeError, KeyError):
                    print("Couldn't determine YOLO model from saved model file")
                    model = input("Enter YOLO model: ")
            except FileNotFoundError:
                raise SystemExit(f"Invalid saved model path: \"{model}\"")
        export_info = ModelExportInfo(model, inp_width, inp_height, export_format, saved_weights)
        exporters.append(YOLOModelExporter(export_info))

    return export_and_save_models(
        exporters,
        quant_types,
        quant_dataset,
        export_dir,
        no_parallel
    )



def main() -> None:
    parser = argparse.ArgumentParser(
    prog=f"python -m pysynap.tools.export.yolo", description=__doc__
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="YOLOv8 or YOLOv9 model(s) (e.g.: yolov9c-seg, yolov8n, yolov8s-pose, ...)",
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        default=f"{os.getcwd()}/models/exported",
        metavar="DIR",
        help="Exported models directory (default: %(default)s)",
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
        help="Dataset to be used for quantization",
    )
    parser.add_argument(
        "--no_parallel",
        action="store_true",
        default=False,
        help="Disable parallel exports. Useful for resource constrained systems",
    )
    args = parser.parse_args()

    export_yolo_models(
        args.models,
        args.input_sizes,
        args.export_formats,
        args.quant_types,
        args.quant_dataset,
        args.export_dir,
        args.no_parallel
    )


if __name__ == "__main__":
    main()
