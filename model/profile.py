"""Profile converted models on a development board with synap_cli"""

import argparse
from pathlib import Path

from model.utils.temp_script import TempScript

# TODO: add ADB support
# TODO: export results as CSV
def profile_models(
    model_names: list[str], models_dir: str, board_ip: str, profile_all: bool
) -> None:
    cp_bench_cmd: str = (
        f"scp model/scripts/benchmark.sh root@{board_ip}:{models_dir}/benchmark.sh"
    )
    profile_cmd: str = f"ssh -T root@{board_ip} << EOF\n"
    profile_cmd += f"cd {models_dir}\nchmod u+x benchmark.sh\n"
    if profile_all:
        profile_cmd += "./benchmark.sh ."
    if not profile_all:
        profile_cmd += "./benchmark.sh " + " ".join(
            f"{model_name}.synap" for model_name in set(model_names)
        )
    profile_cmd += "\ncat results.txt\nEOF\n"
    se = TempScript(cp_bench_cmd, profile_cmd)
    se.run(success_msg=None, error_msg="Profiling failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"python -m src.{Path(__file__).stem}", description=__doc__
    )
    group = parser.add_argument_group(
        "model selection",
        "specify which model(s) to profile in the models directory",
    ).add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--models",
        type=str,
        nargs="+",
        metavar="NAME",
        help="Profile specific model(s)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Profile all models",
    )
    parser.add_argument(
        "--board_ip",
        type=str,
        required=True,
        metavar="ADDR",
        help="Dev board IP address",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="/home/root/models",
        metavar="DIR",
        help="Profile models in this directory (default: %(default)s)",
    )
    args = parser.parse_args()
    profile_models(args.models, args.models_dir, args.board_ip, args.all)
