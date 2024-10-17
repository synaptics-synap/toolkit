#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

import logging
import os
import shutil
import warnings
from logging.handlers import MemoryHandler
from pathlib import Path
import sys
import pathlib
import argparse
import traceback
from tempfile import TemporaryDirectory

import pysynap
from pysynap.converter import HeterogeneousConverter, pytorch_to_onnx
from pysynap.converters import ConversionOptions, TargetSOC
from pysynap.exceptions import SynapError, ConversionError
from pysynap.meta import load_metafile, MetaInfo, NetworkFormat

# Disable init-time info messages from Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# Used in Acuity for creating ruler files
os.environ['SYNAP_VERSION'] = pysynap.version

def main():
    if len(sys.argv) > 1 and sys.argv[1] in ['--version', '-v']:
        print(pysynap.version)
        return

    parser = argparse.ArgumentParser(description="SyNAP model converter " + pysynap.version)
    parser.add_argument('-v', '--version', help='Show version', action='store_true')
    parser.add_argument('--model', help='Model file (.tflite, .pb, .onnx, .torchscript, .prototxt)', required=True)
    parser.add_argument('--weights', help='Weights file', required=False)
    parser.add_argument('--meta', help='Model metafile (.yaml)')
    parser.add_argument('--target', help=f'Target SoC {{{",".join([t.value for t in TargetSOC])}}}', type=str.upper, required=True)
    parser.add_argument('--out-format', help='Output format: {synap,nb}', default='synap', metavar="FMT")
    parser.add_argument('--out-dir', help='Destination directory for SyNAP model', required=True, metavar="DIR")
    parser.add_argument('--profiling', help='Enable by-layer profiling in generated model', action='store_true')
    parser.add_argument('--verbose', help='Verbose mode', action='store_true')
    parser.add_argument('--silent', help='Don\'t show progress state', action='store_true')
    parser.add_argument('--cpu-profiling', help=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--debug', help=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--preserve', help=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--tools-dir', help=argparse.SUPPRESS)
    parser.add_argument('--vssdk-dir', help=argparse.SUPPRESS)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    model_path = Path(args.model).absolute()
    weights_path = Path(args.weights).absolute() if args.weights is not None else None

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    memory_handler = None

    if not args.debug:
        warnings.simplefilter("ignore")

    logging.getLogger("synap").setLevel(logging.INFO)
    logging.getLogger("synap.acuitylib").setLevel(logging.INFO)

    if args.verbose or args.debug:
        if args.debug:
            logging.getLogger("synap").setLevel(logging.DEBUG)
            logging.getLogger("synap.acuitylib").setLevel(logging.DEBUG)

        # when we are debugging or in verbose mode we can directly show all the logs in the console
        logging.getLogger().addHandler(stream_handler)
    else:
        # set the logging that we can collect all log and show them in case there are errors
        # when we are not debugging or in verbose mode we buffer the logs into a memory handler so that we
        # can generate more information in case of crash
        # since we don't install a target this handler will not flush even if the capacity is reached
        memory_handler = MemoryHandler(capacity=100000)
        logging.getLogger().addHandler(memory_handler)

    logger = logging.getLogger("synap")

    # make sure the tools dir is available
    tools_dir = args.tools_dir
    if tools_dir is None:
        # Assume the directory layout of the public toolkit directory
        synap_dir = pathlib.Path(__file__).parent.parent.absolute()
        tools_dir = os.path.join(synap_dir, "toolkit-prebuilts")
        # check if prebuilts is available alongside
        if not os.path.exists(tools_dir):
            tools_dir  = os.path.join(synap_dir, "prebuilts")
    if not os.path.isdir(tools_dir):
        raise ConversionError(f"\n\nCan't find directory for toolkit prebuilts: {tools_dir}\n"
                              "You can use the '--tools-dir' option to set the toolkit prebuilts path "
                              "or install it along the toolkit from:\n\n"
                              "      https://github.com/synaptics-synap/toolkit-prebuilts\n")


    work_dir = out_dir / 'build'

    if work_dir.is_dir():
        shutil.rmtree(work_dir)

    try:
        work_dir.mkdir(parents=True)
    except:
        root_path = '/' + work_dir.parts[1]
        sys.stderr.write(f"Error: failed to create working directory: {work_dir}\n"
                         f"Add  -v {root_path}:{root_path}  to the synap alias"
                         " to make this directory visible inside the container.\n")
        sys.exit(1)

    if args.debug or args.preserve:
        # Also log to file in the working directory
        file_handler = logging.FileHandler(work_dir / "conversion_log.txt")
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

    cache_dir = str((out_dir / "cache").absolute())
    os.makedirs(cache_dir, exist_ok=True)

    try:
        try:
            if args.target == 'VS680A0' or args.target == 'DOLPHIN':
                args.target = 'VS680'
            elif args.target == 'VS640A0' or args.target == 'PLATYPUS':
                args.target = 'VS640'
            target = TargetSOC(args.target)
        except ValueError:
            raise ConversionError('Invalid target specified: ' + args.target + '. Available targets: ' +
                                  ', '.join([t.value for t in TargetSOC]))

        if not model_path.is_file():
            raise ConversionError(f"model not found: {model_path}")

        if weights_path and not weights_path.is_file():
            raise ConversionError(f"model weights file not found: {weights_path}")


        lfs_magic = b"version https://git-lfs.github.com/spec/v1"

        with model_path.open("rb") as fp:
            magic = fp.read(len(lfs_magic))

            if magic == lfs_magic:
                raise ConversionError(f"{model_path} is a git-lfs link file,"
                                              " please ensure the file contents are checked out")

        if args.meta:
            meta = load_metafile(Path(args.meta))
        else:
            meta = MetaInfo()

        if not args.silent:
            print("Converting...\r", end='')

        model_formats = {
            '.tflite': NetworkFormat.TFLITE,
            '.onnx': NetworkFormat.ONNX,
            '.pb': NetworkFormat.TENSORFLOW,
            '.prototxt': NetworkFormat.CAFFE,
            '.torchscript': NetworkFormat.TORCH,
            '.pt': NetworkFormat.TORCH,
            '.pth': NetworkFormat.TORCH
        }

        meta.network_format = model_formats.get(model_path.suffix.lower())
        if meta.network_format == None:
            raise ConversionError(f"unknown network format: {model_path.suffix}")

        if meta.network_format == NetworkFormat.TORCH:
            # Convert torchscript format to ONNX
            model_path = pytorch_to_onnx(model_path, meta, work_dir, args.verbose)
            meta.network_format = NetworkFormat.ONNX

        if args.out_format == 'synap':
            conversion_options = ConversionOptions(verbose=args.verbose, silent=args.silent, debug=args.debug,
                                                   profiling=args.profiling, cpu_profiling=args.cpu_profiling,
                                                   vssdk_dir=args.vssdk_dir, tools_dir=tools_dir,
                                                   cache_dir=cache_dir,
                                                   target=target)

            shc = HeterogeneousConverter(conversion_options)

            # we create the output directory as this is what is specified by the user on the command line
            # the converter will not create it for us as it takes the file name as input
            out_dir.mkdir(exist_ok=True, parents=True)

            shc.convert(model_path, weights_path, meta, out_dir / ('model.synap'), work_dir)

        else:
            if args.out_format == 'ebg':
                print("Warning: 'ebg' format specification is deprecated and will be removed in future versions. Use 'nb' format instead.")
            elif args.out_format != 'nb':
                raise ConversionError(f"unknown output format: {args.out_format}")

            # Here we take all paths to be absolute to make sure things still works when acuitlib does chdir
            tools_dir = os.path.abspath(args.tools_dir) if args.tools_dir is not None else None
            vssdk_dir = os.path.abspath(args.vssdk_dir) if args.vssdk_dir is not None else None
            out_dir = out_dir.absolute()
            work_dir_path = str(work_dir.absolute())
            model_path = model_path.absolute()

            prev_cwd = os.getcwd()

            try:
                # delay import here to prevent slow startup by default
                from pysynap.converters.acuity_converter import Converter
                sc = Converter(args.verbose, args.debug, work_dir_path + "/work", cache_dir)

                net = sc.convert(str(model_path), str(weights_path), meta)

                sc.generate(net, meta, target, str(out_dir), tools_dir, vssdk_dir,
                            True, args.profiling, args.cpu_profiling)
            finally:
                os.chdir(prev_cwd)

        if not args.debug and not args.preserve:
            try:
                shutil.rmtree(work_dir)
            except:
                # Sometimes the above rm fails because some quantization tmp files still open.
                # This happens often when using NFS filesystems.
                logger.info(f"Failed to remove working directory: {work_dir}")

    except Exception as e:
        # SynapError exception and exceptions with no_traceback attribute are supported errors for which we
        # can show a nice error message to the user
        supported_error = isinstance(e, SynapError) or hasattr(e, 'no_traceback')

        if not supported_error or args.debug:
            # we log the error so that we can see the stack trace
            logger.exception("Error while converting: %s", e)

        # in case we are not in verbose or debug mode and the error is an unsupported error that should not happen
        # flush out all logs that were collected in memory to help debugging (including the exception trace
        # we just logged)
        if not supported_error and memory_handler:
            memory_handler.setTarget(stream_handler)
            memory_handler.flush()

        # if the error is a supported error we can show the error message directly to the user
        if supported_error:
            sys.stderr.write(f"Error: {e}\n")

        # we clean up the workdir if the error is supported and we are not in debug/preserve mode.
        # otherwise we tell the user where the work dir is located
        if supported_error and not args.debug and not args.preserve:
            shutil.rmtree(work_dir)
        else:
            sys.stderr.write(f"Temporary files available at : {work_dir}\n")

        sys.exit(1)

    if not args.silent:
        print("Conversion successful")
    sys.exit(0)

if __name__ == "__main__":
    main()
