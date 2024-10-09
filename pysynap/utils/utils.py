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


COLORS_COCO = {
    0: [189, 114, 0],    # Blue
    1: [25, 83, 217],    # Orange
    2: [32, 176, 237],   # Yellow
    3: [142, 47, 126],   # Purple
    4: [48, 172, 119],   # Green
    5: [238, 190, 77],   # Light Blue
    6: [47, 20, 162],    # Dark Red
    7: [77, 77, 77],     # Gray
    8: [153, 153, 153],  # Light Gray
    9: [0, 0, 255],      # Red
    10: [0, 128, 255],   # Orange
    11: [0, 191, 191],   # Olive
    12: [0, 255, 0],     # Green
    13: [255, 0, 0],     # Blue
    14: [255, 0, 170],   # Purple
    15: [0, 85, 85],     # Dark Olive
    16: [0, 170, 85],    # Olive Green
    17: [0, 255, 85],    # Light Green
    18: [0, 85, 170],    # Brown
    19: [0, 170, 170],   # Dark Yellow
    20: [0, 255, 170],   # Lime
    21: [0, 85, 255],    # Orange Red
    22: [0, 170, 255],   # Light Orange
    23: [0, 255, 255],   # Yellow
    24: [128, 85, 0],    # Teal
    25: [128, 170, 0],   # Dark Cyan
    26: [128, 255, 0],   # Aquamarine
    27: [128, 0, 85],    # Indigo
    28: [128, 85, 85],   # Slate Blue
    29: [128, 170, 85],  # Light Sea Green
    30: [128, 255, 85],  # Medium Sea Green
    31: [128, 0, 170],   # Dark Magenta
    32: [128, 85, 170],  # Medium Orchid
    33: [128, 170, 170], # Khaki
    34: [128, 255, 170], # Pale Green
    35: [128, 0, 255],   # Deep Pink
    36: [128, 85, 255],  # Hot Pink
    37: [128, 170, 255], # Light Coral
    38: [128, 255, 255], # Light Yellow
    39: [255, 85, 0],    # Dodger Blue
    40: [255, 170, 0],   # Deep Sky Blue
    41: [255, 255, 0],   # Cyan
    42: [255, 0, 85],    # Medium Blue
    43: [255, 85, 85],   # Slate Blue
    44: [255, 170, 85],  # Steel Blue
    45: [255, 255, 85],  # Light Cyan
    46: [255, 0, 170],   # Purple
    47: [255, 85, 170],  # Medium Purple
    48: [255, 170, 170], # Lavender
    49: [255, 255, 170], # Pale Turquoise
    50: [255, 0, 255],   # Magenta
    51: [255, 85, 255],  # Orchid
    52: [255, 170, 255], # Plum
    53: [255, 255, 255], # White
    54: [0, 0, 85],      # Dark Red
    55: [0, 0, 128],     # Maroon
    56: [0, 0, 170],     # Firebrick
    57: [0, 0, 212],     # Crimson
    58: [0, 0, 255],     # Red
    59: [0, 85, 0],      # Dark Green
    60: [0, 128, 0],     # Green
    61: [0, 170, 0],     # Forest Green
    62: [0, 212, 0],     # Lime Green
    63: [0, 255, 0],     # Lime
    64: [85, 0, 0],      # Midnight Blue
    65: [128, 0, 0],     # Navy
    66: [170, 0, 0],     # Dark Blue
    67: [212, 0, 0],     # Medium Blue
    68: [255, 0, 0],     # Blue
    69: [0, 85, 85],     # Olive Drab
    70: [0, 85, 128],    # Peru
    71: [0, 85, 170],    # Chocolate
    72: [0, 85, 212],    # Dark Orange
    73: [0, 85, 255],    # Orange Red
    74: [0, 128, 85],    # Olive Green
    75: [0, 128, 128],   # Dark Khaki
    76: [0, 128, 170],   # Goldenrod
    77: [0, 128, 212],   # Dark Goldenrod
    78: [0, 128, 255],   # Orange
    79: [0, 170, 85],    # Dark Olive Green
}
