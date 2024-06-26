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

    network = synap.Network()

    if network.load_model(args.model) is not True:
        print("Failed to load model")
        return

    input_data = synap.InputData(args.input)
    if input_data.empty() is True:
        print("Failed to read image data from file")
        return

    if network.inputs[0].assign(input_data.data(), input_data.size()) is not True:
        print("Failed to assign image data to input tensor")
        return

    if network.predict() is not True:
        print("Failed to predict")
        return

    print(network.outputs.size())


if __name__ == "__main__":
    main()
