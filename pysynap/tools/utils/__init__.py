# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

def print_err(msg: str, details: str | None = None) -> None:
    print(f"\033[31mError: {msg}\033[0m", end="")
    if details:
        print(f"\033[31m:\n\t{details}\033[0m")
    else:
        print()