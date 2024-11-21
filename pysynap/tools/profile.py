"""Profile converted models on a development board with synap_cli"""

import argparse
import importlib.resources

from .utils.board_utils import check_cnxn_adb, check_cnxn_ssh
from .utils.temp_script import TempScript


def _get_benchmark_script() -> str:
    with importlib.resources.path("pysynap.utils", "benchmark.sh") as script_path:
        return str(script_path)

def profile_models_ssh(
    model_names: list[str], models_dir: str, board_ip: str | None, profile_all: bool
) -> None:
    if not models_dir.startswith("/"):
        models_dir = "/" + models_dir
    bench_script = _get_benchmark_script()
    cp_bench_cmd = f"scp -o BatchMode=yes -o ConnectTimeout=5 {bench_script} root@{board_ip}:{models_dir}/benchmark.sh"
    profile_cmd = f"ssh -o BatchMode=yes -o ConnectTimeout=5 -T root@{board_ip} << 'EOF'\n"
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


def profile_models_adb(
    model_names: list[str], models_dir: str, serial: str, profile_all: bool
) -> None:
    if not models_dir.startswith("/"):
        models_dir = "/" + models_dir
    bench_script = _get_benchmark_script()
    adb = f"adb -s {serial}" if serial else "adb"
    cp_bench_cmd = f"{adb} push {bench_script} {models_dir}/benchmark.sh > /dev/null"
    chmod_cmd = f"{adb} shell chmod u+x {models_dir}/benchmark.sh"
    if profile_all:
        profile_cmd = f"{adb} shell {models_dir}/benchmark.sh {models_dir}"
    else:
        profile_cmd = f"{adb} shell {models_dir}/benchmark.sh " + " ".join(
            f"{models_dir}/{model_name}.synap" for model_name in set(model_names)
        )
    mv_res_cmd = f"{adb} shell mv /results.txt {models_dir}/results.txt"
    cat_res_cmd = f"{adb} shell cat {models_dir}/results.txt"
    se = TempScript(cp_bench_cmd, chmod_cmd, profile_cmd, mv_res_cmd, cat_res_cmd)
    se.run(success_msg=None, error_msg="Profiling failed")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog=f"python -m pysynap.tools.profile", description=__doc__
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
        "--serial",
        type=str,
        help="Specify serial for ADB, will use first detected device otherwise",
    )
    parser.add_argument(
        "--board_ip",
        type=str,
        metavar="ADDR",
        help="Dev board IP address, for profiling with SSH instead of ADB",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="/home/root/models",
        metavar="DIR",
        help="Profile models in this directory (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.board_ip:
        if check_cnxn_ssh(args.board_ip):
            profile_models_ssh(args.models, args.models_dir, args.board_ip, args.all)
    else:
        if check_cnxn_adb(args.serial):
            profile_models_adb(args.models, args.models_dir, args.serial, args.all)


if __name__ == "__main__":
    main()
