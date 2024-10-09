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

def get_colors_from_json(colors_json: str | os.PathLike, verbose: bool = False):
    """
    Parses [R,G,B] colors from a JSON file specified at @param:colors_json.
    R, G, B must be >= 0 and <= 255, and JSON must have format:
    {
        "class_index": [R,G,B],
        ...
    }
    Logs errors if @param:verbose is set to True.
    """

    def is_valid_color(color):
        return (
            isinstance(color, list) and 
            len(color) == 3 and 
            all(isinstance(c, int) and 0 <= c <= 255 for c in color)
        )

    try:
        with open(colors_json, "r") as f:
            mask_colors = json.load(f)
            mask_colors = {int(k): v for k, v in mask_colors.items()}
            if not(all(is_valid_color(color) for color in mask_colors.values())):
                raise ValueError(f"Erroneous color values in \"{colors_json}\"")
            return mask_colors
    except ValueError as e:
        if verbose:
            print(f"ERROR: Couldn't parse mask colors: {e.args[0]}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        if verbose:
            print(f"ERROR: Invalid mask colors file \"{colors_json}\"")
    return None
