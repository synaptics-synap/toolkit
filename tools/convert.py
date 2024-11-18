"""Convert exported models using the SyNAP toolkit"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from os import getcwd
from pathlib import Path
from sys import exit
from time import sleep

from tools.utils.temp_script import TempScript


def convert_model(
    model_path: Path, convert_dir: Path, target: str, profile: bool
) -> None:
    model_name: str = model_path.stem
    meta_path: Path = model_path.parent / f"{model_name}.yaml"
    out_dir: Path = convert_dir / f"{model_name}"
    conv_cmd: str = (
        "synap_convert"
        + f' --model "{model_path.resolve()}"' 
        + f' --meta "{meta_path.resolve()}"' 
        + f" --target {target}" 
        + f' --out-dir "{out_dir.resolve()}"'
    )
    if profile:
        conv_cmd += " --profiling"
    print(f"Converting {model_name}:\n{conv_cmd}\n")
    se = TempScript(conv_cmd)
    se.run(
        success_msg=f"Converting {model_name}:",
        error_msg=f"Converting {model_name} failed",
    )


def convert_multiple(
    model_paths: list[Path],
    convert_dir: Path,
    target: str,
    profiling: bool,
    no_parallel: bool,
) -> None:
    if no_parallel:
        for model_path in model_paths:
            convert_model(model_path, convert_dir, target, profiling)
    else:
        with ProcessPoolExecutor() as executor:
            for model_path in model_paths:
                executor.submit(
                    convert_model, model_path, convert_dir, target, profiling
                )
                sleep(0.5)  # for concurrency stability


def main() -> None:
    export_dir: Path = Path(args.export_dir)
    convert_dir: Path = Path(args.convert_dir)
    model_paths: list[Path] = []
    args.models = list(set(args.models) if args.models else set())
    if args.models:
        if len(args.models) == 1:
            model_paths.extend(
                [
                    p
                    for p in list(Path(export_dir).glob(args.models[0]))
                    if p.suffix.lstrip(".") in args.export_formats
                ]
            )
        else:
            model_paths.extend([Path(export_dir) / model for model in args.models])
        if not model_paths:
            print(f"No models to convert in {export_dir.resolve()}")
            exit()
        convert_multiple(
            model_paths, convert_dir, args.target, args.profiling, args.no_parallel
        )
        return
    else:
        for fmt in args.export_formats:
            model_paths.extend(list(export_dir.glob(f"*.{fmt}")))
    if not model_paths:
        print(f"No models to convert in {export_dir.resolve()}")
        exit()
    if args.latest:
        latest_model: Path = max(model_paths, key=lambda p: p.stat().st_mtime)
        convert_model(latest_model, convert_dir, args.target, args.profiling)
    else:
        convert_multiple(
            model_paths, convert_dir, args.target, args.profiling, args.no_parallel
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"python -m tools.convert", description=__doc__
    )
    group = parser.add_argument_group(
        "model selection",
        "specify which model(s) to convert from the exported models directory",
    ).add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--models",
        type=str,
        nargs="+",
        metavar="NAME",
        help="Convert specific model(s)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Convert all models",
    )
    group.add_argument(
        "--latest",
        action="store_true",
        help="Convert the most recently exported model",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=["GENERIC", "DVF120", "VS640", "VS680", "SL1620", "SL1640", "SL1680"],
        help="Target SoC",
    )
    parser.add_argument(
        "--export_formats",
        nargs="+",
        default=["tflite", "onnx"],
        metavar="FMT",
        choices=["tflite", "onnx", "pb", "pt"],
        help="Exported model formats to include (default: %(default)s)",
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        default=f"{getcwd()}/models/exported",
        metavar="DIR",
        help="Exported models directory (default: %(default)s)",
    )
    parser.add_argument(
        "--convert_dir",
        type=str,
        default=f"{getcwd()}/models/converted",
        metavar="DIR",
        help="Converted models directory (default: %(default)s)",
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
        help="Do not convert multiple models in parallel. Useful for resource constrained systems",
    )
    args = parser.parse_args()
    main()
