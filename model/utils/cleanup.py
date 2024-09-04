"""Cleans up YOLO training and export files"""

import argparse
import os
import sys
from pathlib import Path
from shutil import rmtree


def cleanup(
    export_formats: list[str],
    remove_exported: bool,
    remove_converted: bool,
    export_dir: str,
    convert_dir: str,
) -> None:
    files: list[Path] = []
    dirs: list[Path] = []
    if (runs_dir := Path(f"./runs")).exists():
        dirs.append(runs_dir)
    for f in Path(".").iterdir():
        if f.is_dir() and "saved_model" in f.name:
            dirs.append(f)
        if f.is_file() and f.suffix.lstrip(".") in [*export_formats, "sh", "npy", "pt"]:
            files.append(f)
    if remove_exported and Path(export_dir).exists():
        for f in Path(export_dir).iterdir():
            if f.is_file() and f.suffix.lstrip(".") in [*export_formats, "yaml"]:
                files.append(f)
    if remove_converted and Path(convert_dir).exists():
        for f in Path(convert_dir).iterdir():
            if f.is_file():
                files.append(f)
            else:
                dirs.append(f)
    
    to_delete: list[Path] = [f.resolve() for f in files + dirs]
    if not to_delete:
        print("No files to clean up")
        sys.exit()
    print("The following files and directories will be deleted:\n")
    print(*to_delete, sep=f"\n")
    if input("\nContinue? (Y/[n]): ") in ("Y", "y"):
        for f in to_delete:
            if f.is_file():
                f.unlink()
            else:
                rmtree(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"python -m src.{Path(__file__).stem}", description=__doc__
    )
    parser.add_argument(
        "--export_formats",
        nargs="+",
        default=["tflite", "onnx"],
        metavar="FMT",
        help="Export model formats (default: %(default)s)",
    )
    parser.add_argument(
        "--exported",
        action="store_true",
        default=False,
        help="Remove exported models",
    )
    parser.add_argument(
        "--converted",
        action="store_true",
        default=False,
        help="Remove converted models",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Remove both exported and converted models",
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
    args = parser.parse_args()
    if args.all:
        args.exported, args.converted = True, True
    cleanup(
        args.export_formats,
        args.exported,
        args.converted,
        args.export_dir,
        args.convert_dir,
    )
