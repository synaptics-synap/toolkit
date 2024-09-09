"""Cleans up YOLO training and export files"""

import argparse
import os
import sys
from pathlib import Path
from shutil import rmtree


def cleanup(to_delete: list[Path], bypass_conf: bool = True) -> None:
    to_delete = [f.resolve() for f in to_delete]
    if not to_delete:
        print("No files to clean up")
        sys.exit()
    print("The following files and directories will be deleted:\n")
    print(*to_delete, sep=f"\n")
    if bypass_conf or (input("\nContinue? (Y/[n]): ") in ("Y", "y")):
        for f in to_delete:
            if f.is_file():
                f.unlink()
            else:
                rmtree(f)


def get_model_convert_files(convert_dir: str) -> list[Path]:
    files: list[Path] = []
    dirs: list[Path] = []
    if Path(convert_dir).exists():
        for f in Path(convert_dir).iterdir():
            if f.is_file():
                files.append(f)
            else:
                dirs.append(f)
    return files


def get_model_export_files(export_dir: str, export_formats: list[str]) -> list[Path]:
    files: list[Path] = []
    if Path(export_dir).exists():
        for f in Path(export_dir).iterdir():
            if f.is_file() and f.suffix.lstrip(".") in [*export_formats, "yaml"]:
                files.append(f)
    return files


def main() -> None:
    to_delete: list[Path] = []
    if args.converted:
        to_delete += get_model_convert_files(args.convert_dir)
    if args.exported:
        to_delete += get_model_export_files(args.export_dir, args.export_formats)
    cleanup(to_delete, args.yes)


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
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        default=False,
        help="Bypass cleanup confirmation prompt"
    )
    args = parser.parse_args()
    if args.all:
        args.exported, args.converted = True, True
    main()
