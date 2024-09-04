import tempfile
import subprocess
import stat
from pathlib import Path

__all__ = [
    "TempScript",
]


class TempScript:

    def __init__(self, *cmds: str) -> None:
        self._cmds = cmds

    def run(
        self,
        *,
        success_msg: str | None,
        error_msg: str | None,
    ) -> None:
        with tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".sh"
        ) as temp_script:
            temp_script.write("#!/bin/bash\n\nset -e\n\n")
            for cmd in self._cmds:
                temp_script.write(cmd + "\n")
            temp_script.flush()  # ensure all commands are written
            temp_script_name: str = temp_script.name

        # make script executable
        Path(temp_script_name).chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)

        try:
            res = subprocess.run(
                [temp_script_name], shell=False, check=True, capture_output=True
            )
        except subprocess.CalledProcessError as e:
            if error_msg:
                print(f"{error_msg} with return code: {e.returncode}")
            if e.stderr:
                print(e.stderr.decode())
        else:
            if success_msg:
                print(success_msg)
            if res.stdout:
                print(res.stdout.decode())
        finally:
            try:
                Path(temp_script_name).unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    pass