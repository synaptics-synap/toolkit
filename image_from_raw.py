#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

#
# Convert an image from raw format to .png or .jpg
#

import os
import cv2
import numpy as np
import argparse
import sys
import os.path


scale_value = 1
bias_value = 0

def to_uint8(img):
    np.clip((img + bias_value) * scale_value, 0, 255, out=img)
    return img.astype(np.uint8)

def image_from_raw(raw_filename, width, height, dst:str, type:str, dtype:str):
    base, ext = os.path.splitext(raw_filename)
    if not type:
        type = ext[1:]
    if dst.startswith(".") and not "/" in dst:
        dst = base + dst

    if type == 'nv15' and not dtype:
        dtype = 'uint10'

    if dtype == 'float' or dtype == 'float32':
        npdtype = np.single
    elif dtype == 'float16':
        npdtype = np.float16
    elif dtype == 'uint8' or dtype == 'uint10' or not dtype:
        npdtype = np.uint8
    else:
        print("Unknown data type:", dtype)
        sys.exit(1)

    raw_data = np.fromfile(raw_filename, dtype=npdtype)
    if dtype == 'uint10':
        # Unpack yuv array from 10 bits to np.uint16
        data16 = np.zeros(raw_data.size * 4 // 5, dtype=np.uint16)
        for i in range(data16.size // 4):
            di = i * 4
            si = i * 5
            data16[di] = (raw_data[si]) + ((raw_data[si + 1] & 0x3) << 8)
            data16[di+1] = (raw_data[si + 1] >> 2) + ((raw_data[si + 2] & 0xf) << 6)
            data16[di+2] = (raw_data[si + 2] >> 4) + ((raw_data[si + 3] & 0x3f) << 4)
            data16[di+3] = (raw_data[si + 3] >> 6) + ((raw_data[si + 4]) << 2)
        # Align all values to MSBit
        raw_data = data16 * 64

    type = type.lower()
    y_size = height * width
    if type == 'rgb':
        rgb = to_uint8(raw_data.reshape(height, width,3))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dst, bgr)
        return
    if type == 'rgbp':
        # RGB planar
        rgbp = to_uint8(raw_data.reshape(3, height, width))
        rgb = np.transpose(rgbp, (1, 2, 0))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dst, bgr)
        return
    elif type == 'bgr':
        bgr = to_uint8(raw_data.reshape(height, width,3))
        cv2.imwrite(dst, bgr)
        return
    elif type == 'bgra':
        bgra = to_uint8(raw_data.reshape(height, width,4))
        cv2.imwrite(dst, bgra[:,:,:-1])
        return
    elif type == 'gray':
        gray = to_uint8(raw_data.reshape(height, width,1))
        cv2.imwrite(dst, gray[:,:,:1])
        return

    yuv = raw_data
    y = yuv[0:y_size].reshape(height, width)
    uv = yuv[y_size:].reshape(height//2, width//2, 2)

    u_up = np.zeros_like(y)
    v_up = np.zeros_like(y)
    h, w = y.shape[0], y.shape[1]
    u_up = cv2.resize(uv[..., 0], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    v_up = cv2.resize(uv[..., 1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if type == 'nv12' or type == 'nv15':
        yuv = cv2.merge([y, u_up, v_up])
    elif type == 'nv21':
        yuv = cv2.merge([y, v_up, u_up])
    else:
        print("Unknown type:", type)
        sys.exit(1)

    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    cv2.imwrite(dst, bgr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--src', help='Source raw image file')
    parser.add_argument('-s', '--size', help='Source image pixel size: WxH')
    parser.add_argument('--format', help='Source image format: nv21, nv12, nv15, rgb, rgbp, bgr, bgra, gray')
    parser.add_argument('--scale', help='Normalization factor applied to each pixel', type=float, default=1)
    parser.add_argument('--bias', help='Normalization bias: out=(img+bias)*scale', type=float, default=0)
    parser.add_argument('--dtype', help='Source image data type: uint8, uint10, float32, float16')
    parser.add_argument('-o', '--dst', help='Destination image file (.png or .jpg)')
    args = parser.parse_args()
    scale_value = args.scale
    bias_value = args.bias

    if not os.path.isfile(args.src):
        print(f"Error: file {args.src} not found.")
        sys.exit(1)

    try:
        if args.size:
            # Get dimension from command line parameter
            w, h = [int(i) for i in args.size.split('x')]
        else:
            # Try to extract dimension from file name (eg: image_rgb@640x480_out.rgb, image_640x480.rgb)
            base_name = os.path.splitext(os.path.split(args.src)[1])[0]
            dims_str = base_name[base_name.rindex('@' if '@' in base_name else '_')+1:].split('_')[0]
            w, h = [int(v) for v in dims_str.split('x')]
    except:
        print("Invalid size specified, use: -s WxH" , file=sys.stderr)
        sys.exit(1)

    image_from_raw(args.src, w, h, args.dst, args.format, args.dtype)

if __name__ == "__main__":
    main()
