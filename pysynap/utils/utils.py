# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

import json
import os
from pathlib import Path


def create_output_dir(path: str | os.PathLike):
    """
    Creates a dir specified at @param:path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(data: dict, output_path: str | os.PathLike):
    """
    Creates a JSON file specified at @param:output_path,
    with @param:data.
    """
    create_output_dir(output_path)
    with open(output_path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)
