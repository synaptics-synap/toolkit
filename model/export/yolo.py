"""Export YOLOv8 and YOLOv9 models to various formats"""

import argparse
from itertools import product
from logging import ERROR
from os import getcwd
from pathlib import Path
from shutil import rmtree
from typing import Any

from ultralytics import YOLO
from ultralytics.utils import LOGGER

from model.export.base import ModelExporter, ModelExportInfo, export_and_save_models
from model.utils.model_info import *

# suppress Ultralytics logging
LOGGER.setLevel(ERROR)


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
    
    def generate_input_metadata(self, model_path: str) -> list[dict]:
        inputs_info: list[dict] = super().generate_input_metadata(model_path)
        return [
            {"scale": 255, "format": "rgb keep_proportions=1", **info}
            for info in inputs_info
        ]
    
    def generate_output_metadata(self, model_path: str) -> list[dict] | None:
        outputs_info: list[dict] = super().generate_output_metadata(model_path)
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


def main() -> None:
    inp_sizes = [
        tuple(int(dim) for dim in size.split("x")) for size in args.input_sizes
    ]
    exporters: list[YOLOModelExporter] = []
    for info in product(args.models, inp_sizes, args.export_formats):
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
    export_and_save_models(
        exporters,
        args.quant_types,
        args.quant_datasets,
        args.export_dir,
        args.no_parallel
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"python -m model.export.yolo", description=__doc__
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="YOLO model (e.g.: yolov9c-seg, yolov8n, yolov8s-pose, ...)",
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        default=f"{getcwd()}/models/exported",
        metavar="DIR",
        help="Exported models directory (default: %(default)s)",
    )
    parser.add_argument(
        "--input_sizes",
        nargs="+",
        default=["640x352"],
        metavar="SIZE",
        help="Input image sizes. Each dimension must be a multiple of 32 (wxh) (default: %(default)s)",
    )
    parser.add_argument(
        "--export_formats",
        nargs="+",
        default=["tflite"],
        metavar="FMT",
        choices=["tflite", "onnx", "pb", "pt"],
        help="Export model formats (default: %(default)s)",
    )
    quant_grp = parser.add_argument_group("optional quantization parameters")
    quant_grp.add_argument(
        "--quant_types",
        nargs="+",
        metavar="TYPE",
        choices=["uint8", "int8", "int16", "float16", "mixed"],
        help="Quantization types to apply",
    )
    quant_grp.add_argument(
        "--quant_datasets",
        type=str,
        metavar="FILE",
        nargs="+",
        help="Dataset(s) to be used for quantization",
    )
    parser.add_argument(
        "--no_parallel",
        action="store_true",
        default=False,
        help="Disable parallel exports. Useful for resource constrained systems",
    )
    args = parser.parse_args()
    main()
