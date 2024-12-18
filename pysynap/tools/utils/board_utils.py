import subprocess

from . import print_err


def check_cnxn_adb(device_id: str | None = None, timeout: int = 5) -> bool:
    try:
        adb_cmd = ["adb"]
        if device_id:
            adb_cmd.extend(["-s", device_id])
        adb_cmd.extend(["shell", "true"])

        subprocess.run(
            adb_cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        if isinstance(e, subprocess.CalledProcessError):
            details = e.stderr.decode().strip().replace("\n", "\n\t")
        else:
            details = f"timed out after {timeout} seconds"
        print_err(f"ADB failed to connect to board{' ' + device_id if device_id else ''}", details)
    except FileNotFoundError:
        print_err("ADB binary not found")
    return False


def check_cnxn_ssh(board_ip: str, timeout: int = 5) -> bool:
    try:
        subprocess.run(
            [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", f"ConnectTimeout={timeout}",
                f"root@{board_ip}", "true"
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        return True
    except subprocess.CalledProcessError as e:
        print_err(f"SSH Failed to connect to board {board_ip}", e.stderr.decode().strip().replace("\n", "\n\t"))
    except FileNotFoundError:
        print_err("SSH binary not found")
    return False
