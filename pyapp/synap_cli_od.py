#!/usr/bin/env python3
# coding=utf-8

import os
import sys
import argparse

sys.path.append(os.environ.get("LD_LIBRARY_PATH"))

import synap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='synap model file path')
    parser.add_argument('-i', '--input', help='Detect image file')
    args = parser.parse_args()

    preprocessor = synap.Preprocessor()
    network = synap.Network()
    detector = synap.Detector()
    od_result = synap.DetectorResult()

    if network.load_model(args.model) is not True:
        print("Failed to load model")
        return

    if detector.init(network.outputs) is not True:
        print("Failed to init detector")
        return

    input_data = synap.InputData(args.input)
    if input_data.empty() is True:
        print("Failed to read image data from file")
        return

    if network.inputs[0].assign(input_data.data(), input_data.size()) is not True:
        print("Failed to assign image data to input tensor")
        return

    # FIXME: cannot work because Tensor Class doesn't support move constructor
    # if preprocessor.assign(network.inputs[0], input_data) is not True:
    #     print("Failed to assign input to tensor")
    #     return

    if network.predict() is not True:
        print("Failed to predict")
        return

    assigned_rect = synap.Rect()
    result = detector.process(network.outputs, assigned_rect)
    print(result.success)

    print(result.items[0])
    print(len(result.items))


if __name__ == "__main__":
    main()
