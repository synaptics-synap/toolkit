import os
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2, copytree
from tempfile import TemporaryDirectory

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from ..utils.model_info import *

__all__ = [
    "ModelExportInfo",
    "ModelExporter",
    "export_and_save_models",
]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

CACHE_ROOT_DIR: str = f"{os.getcwd()}/.export_cache"

@dataclass
class ModelExportInfo:
    model_name: str
    inp_width: int
    inp_height: int
    export_format: str
    saved_weights: Path | None

    @property
    def model_id(self) -> str:
        return f"{self.model_name}_{self.inp_width}x{self.inp_height}"

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

    @staticmethod
    def check_metadata(metadata: dict[str, list | dict]) -> CommentedMap:
        commented_metadata = CommentedMap(metadata)
        for section, data in commented_metadata.items():
            if data is None:
                commented_metadata.yaml_set_comment_before_after_key(
                    section,
                    before=(
                        f"{section.capitalize()} metadata missing\n"
                        "See guide on manually adding metadata: https://synaptics-synap.github.io/doc/manual/working_with_models.html#conversion-metafile"
                    ),
                )
            elif not data:
                del commented_metadata[section]

        return commented_metadata

    def generate_input_metadata(self, model_path: str) -> list[dict] | None:
        model_path = str(model_path)
        if self.model_info.export_format == "onnx":
            return get_onnx_layer_info(model_path, "input")
        elif self.model_info.export_format == "tflite":
            return get_tflite_layer_info(model_path, "input")
        else:
            print(
                f'WARNING: Input metadata generation not available for model type "{self.model_info.export_format}"'
            )
            return None
    
    def generate_output_metadata(self, model_path: str) -> list[dict] | None:
        model_path = str(model_path)
        if self.model_info.export_format == "onnx":
            return get_onnx_layer_info(model_path, "output")
        elif self.model_info.export_format == "tflite":
            return get_tflite_layer_info(model_path, "output")
        else:
            print(
                f'WARNING: Output metadata generation not available for model type "{self.model_info.export_format}"'
            )
            return None
    
    def generate_quant_metadata(self, quant_type: str | None, quant_dataset: str | None) -> dict[str, str]:
        quant_info: dict[str, str] = {}
        if quant_type and quant_type != "float16":
            if not quant_dataset:
                raise ValueError(f"Quantization dataset not provided for quantization type {quant_type}")
            if quant_type in ("uint8", "int8", "int16"):
                quant_info["data_type"] = quant_type
                quant_info["scheme"] = (
                    "asymmetric_affine"
                    if quant_type in ("uint8", "int8")
                    else "dynamic_fixed_point"
                )
            elif quant_type == "mixed":
                quant_info.update({"data_type": {'"*"': "uint8"}})
            quant_info["dataset"] = quant_dataset
        return quant_info
    
    def generate_metadata(
        self, model_path: str, quant_type: str | None = None, quant_dataset: str | None = None
    ) -> dict[str, list | dict | None]:
        metadata: dict[str, list | dict | None] = {}
        metadata["inputs"] = self.generate_input_metadata(model_path)
        metadata["outputs"] = self.generate_output_metadata(model_path)
        metadata["quantization"] = self.generate_quant_metadata(quant_type, quant_dataset)
        return metadata
    
    def save_exported_model(
        self,
        model_path: str,
        export_dir: str,
        metadata: dict[str, list | dict] | None,
        quant_type: str | None,
    ) -> dict[str, Path | None]:
        
        def tr_yaml(yaml: str) -> str:
            return yaml.replace("'", "").replace('"', "'")

        saved_files: dict[str, Path | None] = {"model": None, "yaml": None}
        export_dir: Path = Path(export_dir)
        if not export_dir.exists():
            export_dir.mkdir(exist_ok=True, parents=True)
        model_path: Path = Path(model_path)
        exported_model_path: Path = export_dir / self.model_info.export_filename(quant_type)
        copy2(model_path, exported_model_path)
        if exported_model_path.exists():
            saved_files["model"] = exported_model_path
        # print(f'Exported model "{model_path}" copied to "{export_path}"')
        if metadata:
            commented_metadata = self.check_metadata(metadata)
            if commented_metadata.get("quantization") and quant_type == "mixed":
                metadata["quantization"] = CommentedMap(commented_metadata["quantization"])
                metadata["quantization"].yaml_set_comment_before_after_key("data_type", after="add mixed quantzation layers here")
            exported_yaml_path: Path = export_dir / self.model_info.yaml_filename(quant_type)
            with open(
                exported_yaml_path, "w"
            ) as f:
                yaml = YAML()
                yaml.default_flow_style = False
                yaml.dump(commented_metadata, f, transform=tr_yaml)
            if exported_yaml_path.exists():
                saved_files["yaml"] = exported_yaml_path
        else:
            print(f"Metadata file not generated for {self.model_info}, please create manually")

        return saved_files

    def cleanup_export_files(self) -> None:
        pass


def __run_exporter(
    exporter: ModelExporter,
    export_dir: str,
    quant_types: list[str],
    quant_dataset: str | None = None,
) -> list[Path]:
    model_info: ModelExportInfo = exporter.model_info
    exported_paths: list[Path] = []
    curr_wd = os.getcwd()
    with TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        __fetch_from_cache(model_info.model_id, temp_dir)
        if model_info.saved_weights:
            model_info.saved_weights = Path(copy2(model_info.saved_weights, temp_dir))
        model_path: str = exporter.export_model()
        print(f"Exporting {model_info} ... complete")
        for quant_type in quant_types:
            metadata: dict[str, list | dict] | None = exporter.generate_metadata(
                model_path, quant_type, quant_dataset
            )
            export_filename: str = model_info.export_filename(quant_type)
            if metadata:
                print(
                    f"Generating metadata for {export_filename} ... complete"
                )
            saved_files = exporter.save_exported_model(model_path, export_dir, metadata, quant_type)
            print(
                f"Saving model and metadata for {export_filename} ... complete"
            )
            if (model_path := saved_files.get("model")) and saved_files.get("yaml"):
                exported_paths.append(model_path)
        exporter.cleanup_export_files()
        __update_cache(model_info.model_id, temp_dir)
    os.chdir(curr_wd)
    return exported_paths


def __fetch_from_cache(model_id: str, model_files_dir: str) -> None:
    if (model_cache := Path(CACHE_ROOT_DIR) / model_id).exists():
        for f in model_cache.iterdir():
            if f.is_dir():
                copytree(model_files_dir, f, dirs_exist_ok=True)
            elif f.is_file():
                copy2(f, model_files_dir)


def __update_cache(model_id: str, model_files_dir: str) -> None:
    if not (model_cache := Path(CACHE_ROOT_DIR) / model_id).exists():
        model_cache.mkdir(parents=True)
    copytree(model_files_dir, model_cache, dirs_exist_ok=True)


def export_and_save_models(
    exporters: list[ModelExporter],
    quant_types: list[str] | None,
    quant_dataset: str | None,
    export_dir: str,
    no_parallel: bool,
) -> list[Path]:
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

    exported_paths: list[Path] = []
    if input("Continue? ([y]/n): ") not in ("n", "N"):
        if no_parallel:
            for exporter in exporters:
                exported_paths.extend(
                    __run_exporter(
                        exporter,
                        export_dir,
                        quant_types,
                        quant_dataset
                    )
                )
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        __run_exporter,
                        exporter,
                        export_dir,
                        quant_types,
                        quant_dataset,
                    )
                    for exporter in exporters
                ]
                for future in as_completed(futures):
                    try:
                        exported_paths.extend(future.result())
                    except KeyboardInterrupt as e:
                        raise KeyboardInterrupt from e
                    except Exception as e:
                        print(f"Error: Model export failed: {e}")

    return exported_paths
