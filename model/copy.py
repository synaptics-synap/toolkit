"""Copy converted models to development board"""

import argparse
from os import getcwd
from pathlib import Path
from sys import exit

from model.utils.temp_script import TempScript


# TODO: add ADB support
def copy_models_to_board(
    models_info: dict[str, Path], board_ip: str, copy_dir: str
) -> None:
    copy_cmd: str = ""
    for model_name, model_path in models_info.items():
        mkdir_cmd = f"ssh -T root@{board_ip} \"mkdir -p {copy_dir}\""
        copy_cmd = f"scp {model_path}/model.synap root@{board_ip}:{copy_dir}/{model_name}.synap > /dev/null\n"
        se = TempScript(mkdir_cmd, copy_cmd)
        se.run(success_msg=f'copied "{copy_dir}/{model_name}.synap" to "root@{board_ip}:{copy_dir}/{model_name}.synap"', error_msg="Model copy failed")


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
    models_info: dict[str, Path] = get_models_info(
        Path(args.convert_dir), args.models, args.all, args.latest
    )
    copy_models_to_board(models_info, args.board_ip, args.copy_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"python -m model.copy", description=__doc__
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
        "--board_ip",
        type=str,
        required=True,
        metavar="ADDR",
        help="Dev board IP address",
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
    main()
