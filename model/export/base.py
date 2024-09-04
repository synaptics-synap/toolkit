import os
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory
from time import sleep

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from model.utils.model_info import *

__all__ = [
    "ModelExportInfo",
    "ModelExporter",
    "export_and_save_models",
]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

@dataclass
class ModelExportInfo:
    model_name: str
    inp_width: int
    inp_height: int
    export_format: str
    saved_weights: Path | None

    def export_filename(self, quant_type: str | None) -> str:
        return str(self) + self.__quant_type(quant_type) + f".{self.export_format}"

    def yaml_filename(self, quant_type: str | None) -> str:
        return str(self) + self.__quant_type(quant_type) + ".yaml"
    
    def __quant_type(self, quant_type: str | None) -> str:
        return f"_{quant_type}" if quant_type else ""

    def __repr__(self) -> str:
        return (
            f"{self.model_name}_{self.inp_width}x{self.inp_height}_{self.export_format}"
        )
    

class ModelExporter(ABC):

    def __init__(self, model_info: ModelExportInfo) -> None:
        self.model_info = model_info

    @abstractmethod
    def export_model(self) -> str:
        ...

    def generate_input_metadata(self, model_path: str) -> list[dict]:
        if self.model_info.export_format == "onnx":
            inputs_info = get_onnx_inp_info(model_path)
        elif self.model_info.export_format == "tflite":
            inputs_info = get_tflite_inp_info(model_path)
        else:
            raise ValueError(
                f'Metadata generation not available for model type "{self.model_info.export_format}"'
            )
        return inputs_info
    
    def generate_output_metadata(self, model_path: str) -> list[dict]:
        if self.model_info.export_format == "tflite":
            n_outputs = get_tflite_num_outputs(model_path)
        elif self.model_info.export_format == "onnx":
            n_outputs = get_onnx_num_outputs(model_path)
        else:
            raise ValueError(
                f'Metadata generation not available for model type "{self.model_info.export_format}"'
            )
        return [
            {} for _ in range(n_outputs)
        ]
    
    def generate_quant_metadata(self, quant_type: str | None, quant_datasets: list[str] | None) -> dict[str, str]:
        quant_info: dict[str, str] = {}
        if quant_type and quant_type != "float16":
            if not quant_datasets:
                raise ValueError("Quantization dataset not provided")
            if quant_type in ("uint8", "int8", "int16"):
                quant_info["data_type"] = quant_type
                quant_info["scheme"] = (
                    "asymmetric_affine"
                    if quant_type in ("uint8", "int8")
                    else "dynamic_fixed_point"
                )
            elif quant_type == "mixed":
                quant_info.update({"data_type": {'"*"': "uint8"}})
            quant_info["dataset"] = quant_datasets
        return quant_info
    
    def generate_metadata(
        self, model_path: str, quant_type: str | None = None, quant_datasets: list[str] | None = None
    ) -> dict[str, list | dict] | None:
        try:
            inputs_info: list[dict] = self.generate_input_metadata(model_path)
            outputs_info: list[dict] = self.generate_output_metadata(model_path)
            quant_info: dict[str, str] = self.generate_quant_metadata(quant_type, quant_datasets)
        except ValueError as e:
            print(f"ERROR: Couldn't generate metadata: {e.args[0]}")
            return None
        metadata: dict[str, list | dict] = {}
        metadata["inputs"] = inputs_info
        metadata["outputs"] = outputs_info
        if quant_info:
            metadata["quantization"] = quant_info
        return metadata
    
    def save_exported_model(
        self,
        model_path: str,
        export_dir: str,
        metadata: dict[str, list | dict] | None,
        quant_type: str | None,
    ) -> None:
        
        def tr_yaml(yaml: str) -> str:
            return yaml.replace("'", "").replace('"', "'")

        export_dir: Path = Path(export_dir)
        if not export_dir.exists():
            export_dir.mkdir(exist_ok=True, parents=True)
        model_path: Path = Path(model_path)
        export_path: str = f"{export_dir}/{self.model_info.export_filename(quant_type)}"
        copy2(model_path, export_path)
        # print(f'Exported model "{model_path}" copied to "{export_path}"')
        if metadata:
            if quant_type == "mixed":
                metadata["quantization"] = CommentedMap(metadata["quantization"])
                metadata["quantization"].yaml_set_comment_before_after_key("data_type", after="add mixed quantzation layers here")
            with open(
                f"{export_dir}/{self.model_info.yaml_filename(quant_type)}", "w"
            ) as f:
                yaml = YAML()
                yaml.default_flow_style = False
                yaml.dump(metadata, f, transform=tr_yaml)
        else:
            print(f"Metadata file not generated for {self.model_info}, please create manually")


def __run_exporter(
    exporter: ModelExporter,
    export_dir: str,
    quant_types: list[str],
    quant_dataset: str | None = None,
):
    model_info: ModelExportInfo = exporter.model_info
    curr_wd = os.getcwd()
    with TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        if model_info.saved_weights:
            model_info.saved_weights = Path(copy2(model_info.saved_weights, temp_dir))
        model_path: str = exporter.export_model()
        print(f"Exporting {model_info} ... complete")
        for quant_type in quant_types:
            metadata: dict[str, list | dict] | None = exporter.generate_metadata(
                model_path, quant_type, quant_dataset
            )
            if metadata:
                print(
                    f"Generating metadata for {model_info.export_filename(quant_type)} ... complete"
                )
            exporter.save_exported_model(model_path, export_dir, metadata, quant_type)
            print(
                f"Saving model and metadata for {model_info.export_filename(quant_type)} ... complete"
            )
    os.chdir(curr_wd)


def export_and_save_models(
    exporters: list[ModelExporter],
    quant_types: list[str] | None,
    quant_datasets: list[str] | None,
    export_dir: str,
    no_parallel: bool,
) -> None:
    if quant_types is None:
        quant_types = [None]
    model_export_names: list[str] = [
        exp.model_info.export_filename(quant_type)
        for exp in exporters
        for quant_type in quant_types
    ]
    max_workers = os.cpu_count()
    print("\n===========================================================")
    print("Export Summary:")
    print("===========================================================")
    print(f"Models ({len(model_export_names)}):", end="\n\t")
    print(*model_export_names, sep="\n\t")
    print(f"Export dir         : {export_dir}")
    print(f"Concurrent exports : {'No' if no_parallel else str(max_workers)}")
    print("===========================================================\n")
    if input("Continue? ([y]/n): ") not in ("n", "N"):
        if no_parallel:
            for exporter in exporters:
                __run_exporter(exporter, export_dir, quant_types, quant_datasets)
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for exporter in exporters:
                    executor.submit(
                        __run_exporter,
                        exporter,
                        export_dir,
                        quant_types,
                        quant_datasets,
                    )
                    sleep(0.5)
