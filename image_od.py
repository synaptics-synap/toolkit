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

def image_od(src, dst, json_od_result:str):
    img = cv2.imread(src)
    try:
        od_result = json.loads(json_od_result)
    except:
        print("Error: failed to parse JSON data.")
        sys.exit(1)
    if not 'items' in od_result:
        print("Error: object-detection data not found in the input JSON")
        sys.exit(1)

    print("#   Score  Class   Position        Size  Description     Landmarks")
    for i, detection in enumerate(od_result['items']):
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
    cv2.imwrite(dst, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--src', help='Source image (.png or .jpg)')
    parser.add_argument('-o', '--dst', help='Destination image file')
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
    image_od(args.src, args.dst, od_result)
    

   
