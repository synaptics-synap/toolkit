"""Copy converted models to development board"""

import argparse
from os import getcwd
from pathlib import Path
from sys import exit

from .utils.temp_script import TempScript


def copy_models_to_board(
    models_info: dict[str, Path], serial: str | None, board_ip: str | None, copy_dir: str
) -> None:
    if not copy_dir.startswith("/"):
        copy_dir = "/" + copy_dir
    copy_cmd: str = ""
    for model_name, model_path in models_info.items():
        adb = f"adb -s {serial}" if serial else "adb"
        dest = f"{copy_dir}/{model_name}.synap"
        if board_ip:
            mkdir_cmd = f"ssh -T root@{board_ip} \"mkdir -p {copy_dir}\""
            dest = f"root@{board_ip}:" + dest
            copy_cmd = f"scp {model_path}/model.synap {dest} > /dev/null\n"
        else:
            mkdir_cmd = f"{adb} shell mkdir -p {copy_dir}"
            copy_cmd = f"{adb} push {model_path}/model.synap {dest} > /dev/null\n"
        se = TempScript(mkdir_cmd, copy_cmd)
        se.run(success_msg=f'copied "{model_path}/model.synap" to "{dest}"', error_msg="Model copy failed")


def get_models_info(
    convert_dir: Path, model_names: list[str], get_all: bool, get_latest: bool
) -> dict[str, Path]:
    models_info: dict[str, Path] = {
        d.name: d for d in convert_dir.iterdir() if d.is_dir()
    }
    if not models_info:
        print(f"No models to copy from {convert_dir.resolve()}")
        exit()
    if get_all:
        return models_info
    if get_latest:
        latest_model: Path = max(models_info.values(), key=lambda p: p.stat().st_mtime)
        return {latest_model.name: latest_model}
    model_names = list(set(model_names))
    if len(model_names) == 1:
        return {d.name: d for d in convert_dir.glob(model_names[0]) if d.is_dir()}
    return {
        model_name: model_path
        for model_name, model_path in models_info.items()
        if model_name in model_names
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        prog=f"python -m pysynap.tools.copy", description=__doc__
    )
    group = parser.add_argument_group(
        "model selection",
        "specify which model(s) to copy from the converted models directory",
    ).add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--models",
        type=str,
        nargs="+",
        metavar="NAME",
        help="Copy specific model(s)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Copy all models",
    )
    group.add_argument(
        "--latest",
        action="store_true",
        help="Copy the most recently converted model",
    )
    parser.add_argument(
        "--serial",
        type=str,
        help="Specify serial for ADB, will use first detected device otherwise",
    )
    parser.add_argument(
        "--board_ip",
        type=str,
        metavar="ADDR",
        help="Dev board IP address, for copying with SSH instead of ADB",
    )
    parser.add_argument(
        "--convert_dir",
        type=str,
        default=f"{getcwd()}/models/converted",
        metavar="DIR",
        help="Converted models directory (default: %(default)s)",
    )
    parser.add_argument(
        "--copy_dir",
        type=str,
        default="/home/root/models",
        metavar="DIR",
        help="Copied models directory on board, will be created if it doesn't exist (default: %(default)s)",
    )
    args = parser.parse_args()

    models_info: dict[str, Path] = get_models_info(
        Path(args.convert_dir), args.models, args.all, args.latest
    )
    if not models_info:
        print(f"No models to copy from {Path(args.convert_dir).resolve()}")
        exit()
    copy_models_to_board(models_info, args.serial, args.board_ip, args.copy_dir)


if __name__ == "__main__":
    main()
