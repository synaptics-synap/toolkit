#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

#
# Parse output of object-detection and adds bounding boxes and landmarks to an image.
# The detection output in json format is read form the standard input.
#
# Sample usage: 
#
# adb shell "cd /vendor/firmware/models/object_detection/face/model/yolov5s_face_640x480_onnx_mq; synap_cli_od ../../sample/face_720p.jpg" \
#            | synap_run.sh -i image_od.py -i synap/models/object_detection/face/sample/face_720p.jpg  -o face_od.jpg
# open face_od.jpg
#

import cv2
import argparse
import sys
import json
import random
import os.path
import numpy as np

def get_colors_from_json(colors_json: str, verbose: bool):

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
    except (FileNotFoundError, json.JSONDecodeError):
        if verbose:
            print(f"ERROR: Invalid mask colors file \"{colors_json}\"")
    return None

def get_rand_color():
    return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

def scale_mask(mask: np.ndarray, mask_w: int, mask_h: int, inp_w: int, inp_h: int):
    """Reshape mask into 2D and scale to input image size"""
    mask = mask.reshape((mask_h, mask_w))
    return cv2.resize(mask, (inp_w, inp_h), interpolation=cv2.INTER_LINEAR)

def overlay_mask(mask: np.ndarray, img: np.ndarray, roi: tuple[int, int, int, int], thresh: float = 0, color: list[int] | None = None):
    """Confine mask to bounding box and overlay on img"""
    x, y, dx, dy = roi
    mask[y: y + dy, x: x + dx] = mask[y: y + dy, x: x + dx] > thresh
    colored_mask = np.zeros_like(img)
    colored_mask[mask == 1] = color or get_rand_color()
    return cv2.add(img, colored_mask)

def add_mask(mask: np.ndarray, combined_mask: np.ndarray, roi: tuple[int, int, int, int], thresh: float = 0, color: list[int] | None = None):
    """Confine mask to bounding box and add to combined mask"""
    x, y, dx, dy = roi
    mask[y: y + dy, x: x + dx] = mask[y: y + dy, x: x + dx] > thresh
    combined_mask[mask == 1] = color or get_rand_color()

def image_od(src, dst, json_od_result:str, mask_colors: dict, verbose: bool):
    img = cv2.imread(src)
    inp_h, inp_w, _ = img.shape
    try:
        od_result = json.loads(json_od_result)
    except:
        print("Error: failed to parse JSON data.")
        sys.exit(1)
    if not 'items' in od_result:
        print("Error: object-detection data not found in the input JSON")
        sys.exit(1)
    
    prev_mask = None
    combined_mask = np.zeros_like(img)
    # separate loop to prevent masks from affecting bounding box color
    for i, detection in enumerate(od_result['items']):
        try:
            bb = detection['bounding_box']
            x1 = int(bb['origin']['x'])
            y1 = int(bb['origin']['y'])
            dx = int(bb['size']['x'])
            dy = int(bb['size']['y'])
            ci = detection['class_index']
            if detection['mask']['data']:
                mask_h, mask_w = detection['mask']['height'], detection['mask']['width']
                mask = np.array(detection['mask']['data'], dtype=np.float32)
                if prev_mask is not None and np.array_equal(prev_mask, mask):
                    print("Current mask same as previous mask")
                prev_mask = mask
                mask = scale_mask(mask, mask_w, mask_h, inp_w, inp_h)
                mask_color = mask_colors.get(ci, None) if mask_colors else None
                add_mask(mask, combined_mask, (x1, y1, dx, dy), color=mask_color)
                # individually overlay mask on image and save a copy
                # cv2.imwrite(f'mask_{i} (class {ci}).jpg', overlay_mask(mask, cv2.imread(src), (x1, y1, dx, dy), color=mask_color))
        except KeyError as e:
            if verbose:
                print(f"WARNING: Missing {e} data for detection {i}")
            continue
    # overlay combined mask on image
    combined_mask = np.clip(combined_mask, 0, 255)
    img = cv2.add(img, combined_mask)

    print("#   Score  Class   Position        Size  Description     Landmarks")
    for i, detection in enumerate(od_result['items']):
        try:
            bb = detection['bounding_box']
            x1 = int(bb['origin']['x'])
            y1 = int(bb['origin']['y'])
            dx = int(bb['size']['x'])
            dy = int(bb['size']['y'])
            confidence = detection['confidence']
            ci = detection['class_index']
            lms = detection['landmarks']['points']
            print(f"{i:<5}{confidence:.2f}{ci:>7}  {x1:4},{y1:4}   {dx:4},{dy:4}                  ", end='')
            print(" ".join([f"{lm['x']},{lm['y']}" for lm in lms]))
            color = (0, 255, 128)
            cv2.rectangle(img, (x1, y1), (x1 + dx, y1 + dy), color, 2)
            cv2.putText(img, str(ci) + f": {confidence:.2f}", (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)[0]
            for lm in lms:
                lmx = int(lm['x'])
                lmy = int(lm['y'])
                cv2.rectangle(img, (lmx, lmy), (lmx+2, lmy+2), color, 2)
        except KeyError as e:
            if verbose:
                print(f"WARNING: Missing {e} data for detection {i}")
            continue
    cv2.imwrite(dst, img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--src', help='Source image (.png or .jpg)')
    parser.add_argument('-o', '--dst', help='Destination image file')
    parser.add_argument('--mask_colors', help='JSON file containing segmentation mask colors')
    parser.add_argument('--verbose', action="store_true", default=False, help="Enable verbose logging")
    args = parser.parse_args()
    od_result = sys.stdin.read()

    if not os.path.isfile(args.src):
        print(f"Error: file {args.src} not found.")
        sys.exit(1)

    json_begin = od_result.find('{')
    if json_begin < 0:
        print("Error: JSON data not found in the input.")
        sys.exit(1)
    od_result = od_result[json_begin:]

    mask_colors = get_colors_from_json(args.mask_colors or "utils/colors_coco.json", args.verbose)

    image_od(args.src, args.dst, od_result, mask_colors, args.verbose)

if __name__ == "__main__":
    main()