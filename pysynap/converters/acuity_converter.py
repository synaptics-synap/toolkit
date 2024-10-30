# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

import json
import glob
import os
import shutil
import tempfile
import collections
import subprocess
import numpy as np

import pysynap
from . import ConversionOptions, TargetSOC
from ..converters import ConvertedModel
from ..graph_segmenter import SerializedSubgraph
from ..meta import *
from ..network import DataType
from ..acuitylib_path import *

import acuitylib
import acuitylib.vsi_nn
import acuitylib.converter
import acuitylib.converter.utils
import acuitylib.net_output_meta
import logging
from acuitylib.acuitynet import AcuityNet
from acuitylib.quantization.quantizers.asymmetric_quantizer import AsymmQuantParam


logger = logging.getLogger("synap.acuity_converter")


# Tensor dimensions
class Dimensions:
    def __init__(self, l: acuitylib.layer.AcuityLayer, msg: str):
        self.n = 0
        self.c = 0
        self.h = 0
        self.w = 0
        if len(l.params.shape) == 4:
            layout = l.net.get_org_platform_mode()
            if layout == 'nchw':
                self.n = l.params.shape[0]
                self.c = l.params.shape[1]
                self.h = l.params.shape[2]
                self.w = l.params.shape[3]
            elif layout == 'nhwc':
                self.n = l.params.shape[0]
                self.c = l.params.shape[3]
                self.h = l.params.shape[1]
                self.w = l.params.shape[2]
            else:
                raise ConversionError(f"Unsupported platform layout: {layout}")
        if msg and (self.c == 0 or self.h == 0 or self.w == 0):
            raise ConversionError(f"Shape of layer {l.name} not supported {msg}: {l.params.shape}")


class PreprocAttr:
    def __init__(self, vsi_preproc_type: str, format, update_shape_proc):
        self.vsi_preproc_type = vsi_preproc_type
        self.format = format
        self.update_shape = update_shape_proc

    @staticmethod
    def get(preproc_type: str):
        # When there is a choice we always use NHWC layout since this is the one normally used by TF models
        preproc_attributes = {
            'yuv444': PreprocAttr('IMAGE_YUV444', ['y8', 'u8', 'v8'], _nchw_to_nhwc),
            'yuv420': PreprocAttr('IMAGE_I420', ['y8', 'u8', 'v8'], _nchw_to_nhwc),
            'nv12': PreprocAttr('IMAGE_NV12', ['y8', 'uv8'],
                                lambda i, s: ('nhwc', [1, s[2], s[3] // (1 + (i == 1)), i + 1])),
            'nv21': PreprocAttr('IMAGE_NV21', ['y8', 'vu8'],
                                lambda i, s: ('nhwc', [1, s[2], s[3] // (1 + (i == 1)), i + 1])),
            'gray': PreprocAttr('IMAGE_GRAY', ['y8'], _nchw_to_nhwc),
            'rgb': PreprocAttr('IMAGE_RGB', ['rgb'], lambda i, s: ('nhwc', [1, s[2], s[3] // 3, 3])),
            'bgra': PreprocAttr('IMAGE_BGRA', ['bgra'], lambda i, s: ('nhwc', [1, s[2], s[3] // 4, 4])),
            'rgb888p': PreprocAttr('IMAGE_RGB888_PLANAR', ['rgb'], None),
            'rgb888p3': PreprocAttr('IMAGE_RGB888_PLANAR_SEP', ['r8', 'g8', 'b8'], _nchw_to_nhwc),
            'float32': PreprocAttr('TENSOR', [''], None)
        }
        return preproc_attributes.get(preproc_type)


class Converter:

    def __init__(self, verbose: bool, silent : bool, debug: bool, work_dir: str, cache_dir: str=None):
        self._verbose = verbose
        self._silent = silent
        self._base_work_dir = os.path.dirname(work_dir)

        if cache_dir is None:
            self._cache_dir = self._base_work_dir
        else:
            self._cache_dir = cache_dir

        self._work_dir = work_dir

        self._vsi_nn = acuitylib.vsi_nn.VSInn()
        self._quantization_done = False
        self._framework_data_layout = DataLayout.NCHW
        self._synap_cache_path = os.path.join(self._cache_dir, 'synap_cache.bin')
        self._conversion_info_path = os.path.join(self._cache_dir, 'conversion_info.json')

        # Configure SyNAP tile caching
        os.environ['SYNAP_CACHE_CAPACITY'] = '1000'
        os.environ['SYNAP_CACHE_PATH'] = self._synap_cache_path

        # Acuity logging are in debug mode by default. If needed level must be set in main
        # os.environ['ACUITY_LOG_LEVEL'] = 'DEBUG'  # DEBUG, INFO, WARNING, ERROR, CRITICAL

        # OpenVX logs always enabled
        os.environ['VIV_VX_DEBUG_LEVEL'] = '1'

        # TensorFlow info logs (0) are normally not enabled on the console to avoid confusion
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if debug or not verbose else '1'

        # Network compilation details normally not enabled on the console to avoid confusion
        os.environ['NN_EXT_SHOW_PERF'] = '1' if debug or not verbose else '0'

        # Full VSI logs enabled only for debug
        os.environ['VSI_NN_LOG_LEVEL'] = '5' if debug else '2'
        os.environ['CNN_PERF'] = '1' if debug else '0'

        # Uncomment to disable additional layers at the end (experimental, unsupported)
        # os.environ['VIV_VX_ENABLE_INSERT_CMD'] = '0'
    
    def _show_status(self, msg:str = ''):
        if not self._silent:
            print(msg + ' ' * 20 +'\r', end='')

    def convert(self, model_file: str, weights_file: str | None, meta: MetaInfo) -> AcuityNet:
        shutil.rmtree(os.path.join(self._base_work_dir, "work_nbg_unify"), ignore_errors=True)
        shutil.rmtree(self._work_dir, ignore_errors=True)
        os.makedirs(self._work_dir, exist_ok=True)

        curr_cwd = os.getcwd()

        try:
            os.chdir(self._work_dir)  # entropy.txt file will be created here
            if not model_file:
                raise ConversionError("Model file not specified")
            _, ext = os.path.splitext(os.path.basename(model_file))
            ext = ext.lower()

            if ext == '.prototxt':
                if not weights_file:
                    raise ConversionError("Weights file not specified")
                net = self._vsi_nn.load_caffe(model_file, weights_file)
            elif ext == '.tflite':
                meta.check_inputs_with_shape_have_name()
                self._framework_data_layout = DataLayout.NHWC
                net = self._vsi_nn.load_tflite(
                    model_file,
                    inputs=meta.input_names_str(False),
                    input_size_list=meta.input_shapes_str(False),
                    outputs=meta.output_names_str(False)
                )
            elif ext == '.pb':
                meta.check_inputs_with_shape_have_name()
                self._framework_data_layout = DataLayout.NHWC
                if not weights_file:
                    # Assume tensorflow model
                    net = self._vsi_nn.load_tensorflow(
                        model_file, meta.input_names_str(), meta.input_shapes_str(), meta.output_names_str()
                    )
                else:
                    # Assume caffe2 model
                    self._caffe2_to_onnx(model_file, "caffe2_network", weights_file, None,
                                         os.path.join(self._work_dir, "model.onnx"))

            elif ext == '.onnx':
                meta.check_inputs_with_shape_have_name()
                if meta.input_shapes_str(False) and not meta.input_names_str(False):
                    raise ConversionError("Input names mandatory when input shape specified.")
                net = self._vsi_nn.load_onnx(
                    model_file, meta.input_names_str(False), meta.output_names_str(False), meta.input_shapes_str(False)
                )
            else:
                raise ConversionError(f"Unsupported model file: {model_file}")

            if self._framework_data_layout == DataLayout.NCHW and meta.layout == DataLayout.NHWC:
                raise ConversionError(f"{meta.layout} layout not supported with {ext} models.")
            if meta.layout == DataLayout.DEFAULT:
                meta.layout = self._framework_data_layout
            if meta.quantization:
                net = self._quantize(net, meta)
                if os.path.isfile('entropy.txt'):
                    shutil.copyfile('entropy.txt', os.path.join(self._base_work_dir, "quantization_entropy.txt"))

            il = net.get_input_layers(ign_variable=True)
            if len(meta.inputs) > len(il):
                raise ConversionError(f"Model has {len(il)} inputs but {len(meta.inputs)} inputs specified in meta file")
            ol = net.get_output_layers()
            if len(meta.outputs) > len(ol):
                raise ConversionError(f"Model has {len(ol)} outputs but {len(meta.outputs)} outputs specified in meta file")

            if logger.getEffectiveLevel() <= logging.INFO:
                logger.info("Input layers:" +
                             ", ".join([input.name + ":" + str(input.compute_shape_nhwc()[0].dims) for input in il]))

            # As of Acuity 6.30.6, a separate dataset file is required for each input
            self._vsi_nn.generate_inputmeta(net, separated_database=True, dataset_file=['dataset.txt']*len(il), dataset_type=['TEXT']*len(il))
            self._add_preprocessing(net, meta)

            return net

        finally:
            os.chdir(curr_cwd)


    def generate(self, net: AcuityNet, meta: MetaInfo, target: TargetSOC, out_dir: str, tools_dir: str,
                 vssdk_dir: str, ebg: bool, profiling_mode: bool, cpu_profiling_mode: bool):
        """
        Generate EBG and related json metafile for the given network.
        return the number of inputs in the converted model (may be different from the number of 
        inputs in the original model when preprocessing is applied
        """

        self._show_status("Compiling...")
        os.makedirs(out_dir, exist_ok=True)

        tools_bin_dir = os.path.join(tools_dir, 'bin/x86_64-linux-gcc')
        if not os.path.isdir(tools_bin_dir):
            raise ConversionError(f"Can't find host tool directory: {tools_bin_dir}")

        npu_id, ddr_bw, vssdk_encrypt_subdir, soc_name = self._npu_info(target)
        viv_sdk_dir = tools_dir
        tools_lib_dir = os.path.join(tools_dir, 'lib/x86_64-linux-gcc')
        gen_nbinfo = os.path.join(tools_bin_dir, 'nbinfo')
        gen_ebg = os.path.join(tools_bin_dir, 'synap_cli_nb')
        os.environ['SYNAP_OVXLIB_DIR'] = tools_lib_dir
        # This feature can only work with internal tree with ovxlib static libraries.
        os.environ['SYNAP_ENABLE_CPU_PROFILING'] = "1" if cpu_profiling_mode and internal_tree else "0"
        os.environ['SYNA_SOC'] = soc_name
        # FIXME: We should probably get that from cmake variable at some point: VXK_INSTALL_SUBDIR
        if soc_name.find('640') != -1:
            soc_arch = 'platypus'
        elif soc_name.find('680') != -1:
            soc_arch = 'dolphin'
        else:
            soc_arch = ''
        os.environ['SYNAP_VXK_PATH'] = os.path.join(tools_dir, 'vxk', soc_arch)

        if not os.path.isfile(gen_nbinfo):
            raise ConversionError(f"Can't find nbg info tool: {gen_nbinfo}")
        if not os.path.isfile(gen_ebg):
            raise ConversionError(f"Can't find ebg converter tool: {gen_ebg}")

        # If we quantized the model, save quantization info (for reference only)
        if self._quantization_done:
            self._vsi_nn.save_model_quantize(net, out_dir + "/quantization_info.yaml")

        self._add_postprocessing(net, meta)

        if meta.optimize and ddr_bw >= 0:
            os.environ['NN_EXT_DDR_READ_BW_LIMIT'] = str(ddr_bw)
            os.environ['NN_EXT_DDR_WRITE_BW_LIMIT'] = str(ddr_bw)
            os.environ['NN_EXT_DDR_TOTAL_BW_LIMIT'] = str(ddr_bw)

        self._remove_permute = (self._framework_data_layout == DataLayout.NHWC) and (meta.layout == DataLayout.NCHW)

        acuity_521 = False
        # Set parent of working directory. Here is where temporary directories are generated.
        if acuity_521:
            os.chdir(self._work_dir)
        else:
            os.chdir(self._base_work_dir)

        # Clean synap cache if conversion options changed
        conversion_info = {'npu_id': npu_id, 'synap_version': pysynap.version}
        prev_conversion_info = {}
        if os.path.isfile(self._conversion_info_path):
            prev_conversion_info = json.load(open(self._conversion_info_path))
            json.dump({}, open(self._conversion_info_path, 'w'))
        if prev_conversion_info != conversion_info and os.path.isfile(self._synap_cache_path):
            os.remove(self._synap_cache_path)

        tmp_out_dir_path = Path(self._base_work_dir) / "export_work" / "work"
        tmp_out_dir_path.parent.mkdir(parents=True)

        try:
            self._vsi_nn.export_ovxlib(net,
                                       output_path=str(tmp_out_dir_path) + "/",
                                       optimize=npu_id,
                                       dtype="quantized" if True else "float",  # FIXME
                                       pack_nbg_unify=True,
                                       viv_sdk=viv_sdk_dir,
                                       force_remove_permute=self._remove_permute)
        finally:
            if tmp_out_dir_path.is_dir():
                _copytree(tmp_out_dir_path, self._work_dir)

        if cpu_profiling_mode:
            gen_nbg_path = os.path.join(self._work_dir, "gen_nbg")
            gen_nbg_data = os.path.join(self._work_dir, ".export.data")
            gen_nbg_input = os.path.join(self._work_dir, "input_0_0.tensor")
            gen_nbg_run = [gen_nbg_path, gen_nbg_data, gen_nbg_input]
            # Force enable debugging logs
            os.environ['VIV_VX_DEBUG_LEVEL'] = '1'
            # we just want to profile OpenVX export
            os.environ['VIV_VX_ENABLE_SAVE_NETWORK_BINARY'] = "0"
            logger.info("Generating NBG without cache: %s", gen_nbg_run)

            # Run once without cache
            os.environ['SYNAP_CACHE_CAPACITY'] = "0"
            subprocess.run(gen_nbg_run)
            # Get cpu profiling with cache enabled
            logger.info("Generating NBG with cache: ")
            os.environ['SYNAP_CACHE_CAPACITY'] = "10000"
            subprocess.run(gen_nbg_run)
            gen_nbg_gmon = os.path.join(self._work_dir, "gmon.out")
            grof_report = os.path.join(self._work_dir, "profile.txt")
            gprof_run = ["gprof", gen_nbg_path, gen_nbg_gmon]
            with open(grof_report, "w") as report:
                subprocess.run(gprof_run, stdout=report, check=True)

        shutil.copytree(str(tmp_out_dir_path.parent / (tmp_out_dir_path.name + '_nbg_unify')),
                        self._work_dir + "_nbg_unify")

        # Remember conversion options used
        json.dump(conversion_info, open(self._conversion_info_path, 'w'))

        if acuity_521:
            generated_model_dir = os.path.join(self._work_dir, os.path.basename(self._work_dir) + "_nbg_unify")
        else:
            generated_model_dir = self._work_dir + "_nbg_unify"
        generated_nbg_path = os.path.join(generated_model_dir, 'network_binary.nb')
        generated_ebg_path = os.path.join(generated_model_dir, 'model.ebg')
        generated_encrypted_path = os.path.join(generated_model_dir, 'model.ebge')
        destination_bg_path = os.path.join(out_dir, 'model.nb')
        destination_bginfo_path = os.path.join(out_dir, 'model_info.txt')
        destination_bginfo_path_old = os.path.join(out_dir, 'model.info.txt')
        if os.path.isfile(destination_bginfo_path_old):
            os.remove(destination_bginfo_path_old)

        # Generate network info
        with open(destination_bginfo_path, "w") as infofile:
            subprocess.run([gen_nbinfo, '-a', generated_nbg_path], stdout=infofile, check=True)

        # Generate anchors rounded to 6 decimals (more decimals are useless)
        anchors_string = '[]'
        if hasattr(net, 'anchors') and net.anchors:
            anchors_string = json.dumps([round(a, 6) for a in net.anchors], separators=(',', ':'))
            with open(os.path.join(self._work_dir, 'anchors.json'), 'w') as f:
                f.write(anchors_string)
                logger.info("Model anchors generated in: %s", f.name)
        for out in meta.outputs:
            out.data_format = out.data_format.replace('${ANCHORS}', anchors_string)

        in_security_attr = self._create_metafile(net, generated_model_dir + '/nbg_meta.json', out_dir + '/model.json',
                                                 meta)

        generated_model = generated_nbg_path
        if ebg:
            # Convert network to ebg
            gen_ebg_command = [gen_ebg, '--nb', generated_model, '--to-ebg', generated_ebg_path]
            if profiling_mode:
                logger.info("Profiling mode enabled")
                gen_ebg_command.append('--profile')

            p = subprocess.Popen(gen_ebg_command, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)

            for line in p.stdout:
                logger.debug("%s: %s", gen_ebg,
                             line.decode('utf-8', 'backslashreplace').strip())

            p.wait()

            if p.returncode != 0:
                raise SynapError("Error while converting nbg to ebg")

            generated_model = generated_ebg_path

            if meta.security:
                if not vssdk_encrypt_subdir:
                    raise ConversionError("Secure models not supported by the target")
                enc_tool = 'genx_img'
                if vssdk_dir:
                    # Use the tool from vssdk directory
                    if not os.path.isdir(vssdk_dir):
                        raise ConversionError(f"Cannot find vssdk directory: {vssdk_dir}")
                    enc_tool = os.path.join(vssdk_dir, 'factory', 'scripts', vssdk_encrypt_subdir, 'npu', 'bin', enc_tool)
                    if not os.path.isfile(enc_tool):
                        raise ConversionError(f"Can't find encryption tool: {enc_tool}")
                self._encrypt_model(net, generated_model, generated_encrypted_path, meta, in_security_attr, enc_tool)
                generated_model = generated_encrypted_path

        shutil.copy(generated_model, destination_bg_path)
        self._show_status()
        return len(in_security_attr)

    def _quantize(self, net: AcuityNet, meta: MetaInfo) -> AcuityNet:
        q = meta.quantization

        # Check that the number of inputs and the number of quantization datasets match
        input_layers = self._get_sorted_input_layers(net, meta)
        if len(input_layers) != len(meta.inputs):
            raise ConversionError(f"model has {len(input_layers)} inputs but {len(meta.inputs)} quantization dataset specified in meta file")
        
        # Generate random quantization data files if required
        for i in range(len(q.dataset)):
            if q.dataset[i] == 'random':
                q.dataset[i] = os.path.join(self._work_dir, f'quantization_dataset_{i}.npy')
                logger.info("Generating random quantization data for input ", i)
                np.save(q.dataset[i], np.random.rand(*input_layers[i].params.shape))

        META_TO_ACUITY_ALGORITHM = {
            pysynap.meta.QuantizerAlgorithm.STANDARD:
                acuitylib.quantization.types.QuantizerAlgorithm.NORMAL,
            pysynap.meta.QuantizerAlgorithm.MOVING_AVERAGE:
                acuitylib.quantization.types.QuantizerAlgorithm.MOVING_AVERAGE,
            pysynap.meta.QuantizerAlgorithm.KL_DIVERGENCE:
                acuitylib.quantization.types.QuantizerAlgorithm.KL_DIVERGENCE,
            pysynap.meta.QuantizerAlgorithm.AUTO:
                acuitylib.quantization.types.QuantizerAlgorithm.AUTO,
        }

        acuity_algorithm = META_TO_ACUITY_ALGORITHM[q.algorithm]

        dataset_file = os.path.join(self._work_dir, 'quantization_dataset.txt')
        dataset_size = self._create_quantization_dataset(meta, dataset_file)
        self._vsi_nn.set_database(net, dataset_file, dataset_type='TEXT')

        if isinstance(q.data_type, dict):
            quantization_data_type = None
            bytes = [DataType.INT8, DataType.UINT8]
            for spec, dt in q.data_type.items():
                if dt in bytes:
                    # 8bits quantization if any has precedence over 16bits
                    if quantization_data_type in bytes and quantization_data_type != dt:
                        raise ConversionError(
                            f"mixed quantization with {' and '.join([t.value for t in bytes])} not allowed")
                    quantization_data_type = dt
                elif not quantization_data_type and spec == '*':
                    quantization_data_type = dt
            mixed_quantization = not all(dt == quantization_data_type for dt in q.data_type.values())
        else:
            quantization_data_type = q.data_type
            mixed_quantization = False
        quantization_data_type = quantization_data_type if quantization_data_type else DataType.UINT8
        quantization_scheme = q.scheme
        if not quantization_scheme or quantization_scheme == 'default':
            quantization_scheme = 'asymmetric_affine' if quantization_data_type == DataType.UINT8 else 'dynamic_fixed_point'

        logger.info("Quantization scheme: %s",  quantization_scheme)

        if mixed_quantization:
            logger.info("Quantization data_type: %s mixed", quantization_data_type.value)
        else:
            logger.info("Quantization data_type: %s", quantization_data_type.value)

        logger.info(f"Quantization mode: {q.mode}")
        logger.info(f"Quantization algorithm: {q.algorithm.value}")
        logger.info(f"Quantization dataset: {q.dataset}")
        logger.info(f"Quantization steps: {dataset_size}")

        quantize_all_layers = False
        if q.mode == 'full':
            quantize_all_layers = True
        elif q.mode and q.mode != 'standard':
            raise ConversionError(f"Invalid quantization mode: {q.mode}")

        preprocessing_options = {}
        for in_index, l in enumerate(input_layers):
            input = meta.inputs[in_index]

            if input.data_format and 'keep_proportions=0' in input.data_format:
                # TODO: use acuity index here to support multi-input networks with custom input order
                net.get_input_meta().databases[0].ports[in_index].fitting = 'fill'

            has_numpy_samples = q.dataset[in_index].lower().endswith('.npy')
            if (l.params.shape is None or len(l.params.shape) != 4) and not has_numpy_samples:
                raise ConversionError(f".npy quantization file(s) required for input layer '{l.name}' with shape: {l.params.shape}")

            prep_opts = {
                'reverse_channel': input.data_format.lower().startswith('bgr')  # TODO: FIXME!
            }

            if input.shape is not None:
                if not isinstance(input.shape, list):
                    raise ConversionError(f"invalid shape provided for input: {l.name}: {input.shape}")
                prep_opts['shape'] = input.shape

            if input.means:
                dim = Dimensions(l, "when 'means' is specified")
                input_means = input.get_means(dim.c)
                if has_numpy_samples and any(m for m in input_means):
                    logger.warning(f"Note: input means not applied to NumPy quantization files")
                prep_opts['mean'] = input_means

            if input.scale:
                if has_numpy_samples and input.scale != 1:
                    logger.warning(f"Note: input scale not applied to NumPy quantization files")
                prep_opts['scale'] = 1 / input.scale

            preprocessing_options[l.lid] = prep_opts

        self._vsi_nn.set_preprocess(net, preprocessing_options, set_by_lid=True)

        # If we set this flag the algorithm will be automatically forced to kl_divergence,
        # so enable it only if kl_divergence has been selected.
        compute_entropy = acuity_algorithm == acuitylib.quantization.types.QuantizerAlgorithm.KL_DIVERGENCE

        q_iter_base = 0
        q_iter_total = dataset_size * 2 if mixed_quantization else dataset_size
        quantization_progress= lambda i: self._show_status(f"Quantizing... {(i+q_iter_base)*100 // q_iter_total}%")
        q_options = { **q.options, 'quantization_progress': quantization_progress }

        qnet = self._vsi_nn.quantize(net,
                                     algorithm=acuity_algorithm,
                                     quantizer=quantization_scheme,
                                     qtype=quantization_data_type.value,
                                     hybrid=False,
                                     rebuild=not quantize_all_layers,
                                     rebuild_all=quantize_all_layers,
                                     iterations=dataset_size,
                                     compute_entropy=compute_entropy,
                                     **q_options
                                     )

        # Note: in the generated qnet, input layer names containing special chars are modified (e.g ':' to '/')
        # Apply the same changes to our names to avoid mismatch
        for lyr in meta.inputs:
            if lyr.name:
                lyr.name = acuitylib.converter.utils.valid_name(lyr.name)

        if mixed_quantization:
            # Quantize the model again, specifying for which layer(s) we want a different data type
            requantize_spec = {}
            if '*' in q.data_type and q.data_type['*'] != quantization_data_type:
                # This is the default type, so process it before specific type selection(s)
                for lid in qnet.get_layers().keys():
                    requantize_spec[lid] = q.data_type['*']
            for layer_spec, layer_type in q.data_type.items():
                lids = []
                if layer_spec == 'INPUTS':
                    lids = [l.lid for l in qnet.get_input_layers(ign_variable=True)]
                elif layer_spec.endswith('...'):
                    lids = _get_nodes_after(qnet, [_layer_name_to_lid(qnet, layer_spec[:-3])])
                elif layer_spec != '*':
                    lids = [_layer_name_to_lid(qnet, layer_spec)]
                if layer_type == quantization_data_type:
                    for lid in lids:
                        del requantize_spec[lid]
                else:
                    for lid in lids:
                        requantize_spec[lid] = layer_type

            if logger.getEffectiveLevel() >= logging.INFO:
                logger.info(f"Requantizing layers: {len(requantize_spec)}/{len(qnet.get_layers().keys())}")
            if not requantize_spec:
                raise ConversionError(f"No layers to requantize, please check layer data_type specification")

            q_table = qnet.dump_quantize_table()
            q_table['customized_quantize_layers'] = {lid: _hybrid_type(dtype) for lid, dtype in requantize_spec.items()}
            qnet.tensor_mgr.quantize_tab.check_hybrid_dtype_validity()
            qnet.set_quantize_table(q_table)

            q_iter_base = dataset_size
            qnet = self._vsi_nn.quantize(qnet,
                                         algorithm=acuity_algorithm,
                                         quantizer=quantization_scheme,
                                         qtype=quantization_data_type.value,
                                         hybrid=True,
                                         rebuild=False,
                                         rebuild_all=False,
                                         iterations=dataset_size,
                                         compute_entropy=compute_entropy,
                                         **q_options
                                         )

        self._show_status()
        self._quantization_done = True
        return qnet

    # Get list of input layers in the order specified by the user
    @staticmethod
    def _get_sorted_input_layers(net: AcuityNet, meta: MetaInfo):
        in_layers = net.get_input_layers(ign_variable=True)
        if len(in_layers) <= 1:
            # Nothing to sort
            return in_layers
        if meta.input_names_str(False):
            try:
                if hasattr(in_layers[0], 'original_name'):
                    in_layers = [next(l for l in in_layers if l.original_name ==  lyr.name) for lyr in meta.inputs]
                else:
                    in_layers = [next(l for l in in_layers if l.name ==  lyr.name) for lyr in meta.inputs]
            except:
                input_names = [getattr(l, 'original_name', l.name) for l in in_layers]
                raise ConversionError(
                    f"No matching input layers for {[lyr.name for lyr in meta.inputs]} in {input_names}")
        return in_layers

    @staticmethod
    def _create_quantization_dataset(meta: MetaInfo, dataset_file: str) -> int:
        # Expand each dataset glob (sorted to ensure the file order will not change)
        dataset = [sorted(glob.glob(d)) for d in meta.quantization.dataset]
        for d in dataset:
            for filename in d:
                if ' ' in filename:
                    raise ConversionError("Spaces not allowed in quantization dataset: " + filename)

        # Check all expanded datasets have the same number of files
        dataset_lengths = set([len(d) for d in dataset])
        if len(dataset_lengths) > 1:
            raise ConversionError("Quantization datasets must have the same number of files")
        dataset_len = dataset_lengths.pop()
        if dataset_len == 0:
            raise ConversionError("Quantization dataset is empty")

        # Generate dataset file in acuity format
        with open(dataset_file, "w") as f:
            for i in range(dataset_len):
                print(" ".join([d[i] for d in dataset]), file=f)

        return dataset_len

    def _add_preprocessing(self, net: AcuityNet, meta: MetaInfo):
        if not meta.inputs:
            return

        preprocessing_options = {}
        for in_index, l in enumerate(self._get_sorted_input_layers(net, meta)):
            if in_index >= len(meta.inputs):
                break
            input = meta.inputs[in_index]

            if input.preproc and input.preproc.preproc_type and input.preproc.preproc_type != 'none':
                preproc = input.preproc
                preproc_params = {
                    'add_preproc_node': True
                }
                preproc = input.preproc
                preproc_attr = PreprocAttr.get(preproc.preproc_type)
                if not preproc_attr:
                    raise ConversionError(f"Unknown preprocessing type: {preproc.preproc_type}")
                preproc_params['preproc_type'] = preproc_attr.vsi_preproc_type
                if preproc.size:
                    preproc_params['preproc_image_size'] = preproc.size
                    if preproc.preproc_type == 'float32':
                        raise ConversionError(
                            f"Preprocessing type '{preproc.preproc_type}' doesn't support 'size' field")

                data_is_bgr = preproc.preproc_type == 'bgra'
                tensor_is_bgr = input.data_format.lower().startswith('bgr')
                prep_opts = {}
                prep_opts['reverse_channel'] = data_is_bgr != tensor_is_bgr

                if preproc.crop_rect is not None:
                    if not isinstance(preproc.crop_rect, list):
                        raise ConversionError(
                            f"invalid prepro crop_rec provided for input: {l.name}: {preproc.crop_rect}")
                    preproc_params['preproc_crop'] = {
                        'enable_preproc_crop': True,
                        'crop_rect': preproc.crop_rect
                    }

                qp = l.get_output(0).quant_param
                dim = Dimensions(l, 'in preprocessing')
                input.tensor_h = dim.h
                input.tensor_w = dim.w
                if input.means or input.scale:
                    # Take mean/scale from input spec. This is the most accurate way, and these
                    # parameters are mandatory if we quantized the model ourselves.
                    # After they have been applied remove them so we will not apply them again in
                    # the generated json file (float32 is special since it doesn't support mean and scale).
                    if preproc.preproc_type != 'float32':
                        if input.means:
                            prep_opts['mean'] = input.get_means(dim.c)
                            input.means = None
                        if input.scale:
                            prep_opts['scale'] = 1 / input.scale
                            input.scale = None
                elif isinstance(qp, AsymmQuantParam):
                    # Automatically extract quantization parameters from acuity input tensor
                    # (only asymmetric_affine quantization is supported at the moment)
                    logger.info(f"Inferring preprocessing quantization for input %s from model", l.name)
                    prep_opts['mean'] = [qp.zero_point] * dim.c
                    prep_opts['scale'] = float(qp.scale)
                else:
                    if qp is not None:
                        raise ConversionError(
                            f"mean and scale not specified for input {l.name} (auto-detection for {qp.name} quantization not supported preprocessing)")
                # print(f"Preprocessing normalization for input {l.name}: scale: {str(prep_opts.get('scale'))}, mean: {prep_opts.get('mean')}")

                prep_opts['preproc_node_params'] = preproc_params

                preprocessing_options[l.lid] = prep_opts

        if preprocessing_options:
            if meta.layout != self._framework_data_layout:
                raise ConversionError(f"Preprocessing is not compatible with explicit layout: " + meta.layout.value)
            self._vsi_nn.set_preprocess(net, preprocessing_options, set_by_lid=True)
            if logger.getEffectiveLevel() >= logging.DEBUG:
                logger.debug("Inputmeta: %s", json.dumps(self._vsi_nn.get_inputmeta(net)))

    def _add_postprocessing(self, net: AcuityNet, meta: MetaInfo):
        if not meta.dequantize_outputs and not any(out.dequantize for out in meta.outputs):
            return None

        postprocess_list = []
        out_layers = net.get_output_layers()
        # Layer names from ONNX and .PB are not the actual names, skip reordering in that case
        if meta.output_names_str(False) and net.get_org_platform() not in ['onnx', 'tensorflow']:
            # Get list of output layers in the order specified by the user
            selected_outputs = []
            for lyr in meta.outputs:
                try:
                    if hasattr(out_layers[0], 'original_name'):
                        selected_outputs.append(next(l for l in out_layers if l.original_name == lyr.name))
                    else:
                        selected_outputs.append(next(l for l in out_layers if l.name == lyr.name))
                except:
                    output_names = [getattr(l, 'original_name', l.name) for l in net.get_output_layers()]
                    raise ConversionError(f"Specified output: {lyr.name} not in model outputs: "
                                          f"{output_names}")
            out_layers = selected_outputs
        for out_index, l in enumerate(out_layers):
            if meta.dequantize_outputs or out_index < len(meta.outputs) and meta.outputs[out_index].dequantize:
                postprocess_list.append([l.lid, [{
                    'add_postproc_node': True,
                    'perm': list(range(len(l.get_output().shape.dims))),
                    'force_float32': True
                }]])
        if postprocess_list:
            self._vsi_nn.set_app_postprocess(net, postprocess_list, set_by_lid=True)

    # The input tensors in the generated json file don't have a 1-to-1 correspondence
    # with the model inputs when preprocessing is enabled. This is because:
    #  - some preprocessing types (eg. nv12) generate multiple input tensors
    #  - when cropping is enabled 4 additional scalar inputs are added to specify the crop_rect
    # This method will compute the association and for the specified json input return:
    #  - the meta input specification if present
    #  - the preprocessing type if present
    #  - the relative index inside the preprocessing type
    @staticmethod
    def _get_meta_info(meta: MetaInfo, json_inputs: dict, json_input_ix: int):
        if not meta.inputs:
            return (None, None, None)
        json_input_cnt = 0
        json_inputs = list(json_inputs.values())
        for meta_in in meta.inputs:
            preproc_type = meta_in.preproc.preproc_type if meta_in.preproc else None
            preproc_attr = PreprocAttr.get(preproc_type)
            actual_input_count = len(preproc_attr.format) if preproc_attr else 1
            if json_input_ix >= json_input_cnt and json_input_ix < json_input_cnt + actual_input_count:
                return (meta_in, preproc_type, json_input_ix - json_input_cnt)
            json_input_cnt += actual_input_count
            base_cnt = json_input_cnt
            while json_input_cnt < len(json_inputs) and json_inputs[json_input_cnt].get("type") == "scalar":
                if json_input_cnt == json_input_ix:
                    return (None, None, json_input_cnt - base_cnt)
                json_input_cnt += 1
        return (None, None, None)

    # Add additional information to vsi json file:
    # - the 'data_format' for each input and output tensor (for example 'rgb', 'bgr')
    # - the 'delegate' to select npu inference
    # Fix the 'format' attribute which is incorrect when the 'remove_permute' option or preprocessing is used
    # return a list with the security attributes for the inputs
    def _create_metafile(self, net: AcuityNet, vsi_meta_path: str, synap_meta_path: str, meta: MetaInfo):
        in_layers = self._get_sorted_input_layers(net, meta)
        meta_json = json.load(open(vsi_meta_path), object_pairs_hook=collections.OrderedDict)
        meta_json['secure'] = meta.security is not None
        meta_json['delegate'] = 'npu'
        input_tensor_h = input_tensor_w = 0
        for i, (name, info) in enumerate(meta_json['Inputs'].items()):
            input_meta, preproc_type, relative_index = self._get_meta_info(meta, meta_json['Inputs'], i)
            preproc_attr = PreprocAttr.get(preproc_type)
            item_format = input_meta.data_format if input_meta else None
            if not preproc_attr and not item_format and relative_index is not None and info.get("type") == "scalar":
                # This is a crop tensor. Add size of preceding network tensor in width,height order
                if relative_index == 0:
                    item_format = f"tensor_dim={input_tensor_w}"
                elif relative_index == 1:
                    item_format = f"tensor_dim={input_tensor_h}"
            info['data_format'] = item_format if item_format else meta.input_format
            if input_meta:
                if input_meta.scale:
                    info['scale'] = input_meta.scale
                if input_meta.means:
                    dim = Dimensions(in_layers[i], None)
                    info['mean'] = input_meta.get_means(dim.c if dim.c else 1)
                if input_meta.tensor_h:
                    input_tensor_h = input_meta.tensor_h
                if input_meta.tensor_w:
                    input_tensor_w = input_meta.tensor_w
            if meta.security:
                info['security'] = input_meta.security if input_meta else DEFAULT_INPUT_SECURITY
            if self._remove_permute:
                # Note: if preprocessing is enabled this will be overwritten again here below
                info['format'] = 'nchw'
            if preproc_attr:
                info['data_format'] = (preproc_attr.format[relative_index] + ' ' + info['data_format']).rstrip()
                if preproc_attr.update_shape:
                    info['format'], info["shape"] = preproc_attr.update_shape(relative_index, info["shape"])
                    if preproc_type == 'gray' and info["shape"][3] != 1:
                        raise ConversionError(
                            f"Preprocessing '{preproc_type}' can only be used when the input tensor has 1 channel")
                if preproc_type == 'float32':
                    # Acuity wrongly indicates float16 as data type which is actually float32
                    info['dtype'] = 'float32'
                    if 'quantizer' in info:
                        del info['quantizer']
                    if 'quantize' in info:
                        del info['quantize']

        for i, (name, info) in enumerate(meta_json['Outputs'].items()):
            item_format = meta.outputs[i].data_format if i < len(meta.outputs) else None
            info['data_format'] = item_format if item_format else meta.output_format
            if meta.security:
                info['security'] = meta.outputs[i].security if i < len(meta.outputs) else DEFAULT_OUTPUT_SECURITY
            if self._remove_permute:
                info['format'] = 'nchw'
        with open(synap_meta_path, 'w') as f:
            json.dump(meta_json, f, indent=4)
        return [input.get('security', '') for input in list(meta_json['Inputs'].values())]

    def _encrypt_model(self, net: AcuityNet, clear_path: str, encrypted_path: str, meta: MetaInfo, in_security: list,
                       enc_tool: str):
        self._show_status("Encrypting...")
        encrypt_model_tool = os.path.join(toolkit_dir, 'encrypt-model-ebg.py')

        # Create encryption json metadata file from conversion meta info
        encryption_meta_path = Path(self._work_dir) / "encryption.json"
        default_in_security = [DEFAULT_INPUT_SECURITY] * len(net.get_input_layers(ign_variable=True))
        default_out_security = [DEFAULT_OUTPUT_SECURITY] * len(net.get_output_layers())
        with encryption_meta_path.open("w") as fp:
            json.dump({
                # The actual inputs are not 1-to-1 with those specified in the metafile since preprocessing
                # can add addtional inputs, so we have to use the security attributes received as a parameter
                "inputs": in_security,
                "outputs": [o.security for o in meta.outputs] if meta.outputs else default_out_security
            }, fp)

        run_command(' '.join([encrypt_model_tool,
                        '--enc-tool', enc_tool,
                        '--security-policy', str(encryption_meta_path),
                        '--encryption-key', str(meta.security.encryption_key),
                        '--signature-key', str(meta.security.signature_key),
                        '--model-certificate', str(meta.security.model_certificate),
                        '--vendor-certificate', str(meta.security.vendor_certificate),
                        str(clear_path),
                        str(encrypted_path)]))
        self._show_status()

    @staticmethod
    def _caffe2_to_onnx(caffe2_net,
                        caffe2_net_name,
                        caffe2_init_net,
                        value_info,
                        output_file):
        # https://caffe2.ai/doxygen-python/html/conversion_8py_source.html

        # Import caffe2 ony when needed to avoid spurious annoying warning messages about no GPU
        from caffe2.proto import caffe2_pb2
        from caffe2.python.onnx.backend import Caffe2Backend as c2
        import caffe2.python.onnx.frontend as c2_onnx

        c2_net_proto = caffe2_pb2.NetDef()
        with open(caffe2_net, 'rb') as f:
            c2_net_proto.ParseFromString(f.read())
        if not c2_net_proto.name and not caffe2_net_name:
            raise ConversionError('The input caffe2 net does not have name')
        c2_net_proto.name = caffe2_net_name or c2_net_proto.name
        if caffe2_init_net:
            c2_init_net_proto = caffe2_pb2.NetDef()
            with open(caffe2_init_net, 'rb') as f:
                c2_init_net_proto.ParseFromString(f.read())
            c2_init_net_proto.name = '{}_init'.format(caffe2_net_name)
        else:
            c2_init_net_proto = None

        if value_info:
            value_info = json.loads(value_info)

        onnx_model = c2_onnx.caffe2_net_to_onnx_model(
            predict_net=c2_net_proto,
            init_net=c2_init_net_proto,
            value_info=value_info)

        with open(output_file, 'wb') as f:
            f.write(onnx_model.SerializeToString())

    @staticmethod
    def _npu_info(target_soc: TargetSOC):
        if target_soc in [TargetSOC.SL1680, TargetSOC.VS680]:
            return ("VIP9000NANOQI_PLUS_PID0XC1", 22, 'vs680', 'VS680A0')
        elif target_soc in [TargetSOC.SL1640, TargetSOC.VS640]:
            return ("VIP9000NANOSI_PID0XC2", 10, 'vs640', 'VS640A0')
        raise ConversionError(f"Unknown target: {target_soc}")


#
# Utility functions
#


# Implements copytree with dirs_exist_ok=True (available only from python 3.8)
def _copytree(src, dst):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)


# Convert shape from nchw to nhwc
def _nchw_to_nhwc(i: int, s):
    return ('nhwc', [1, s[2], s[3], s[1]])


# Get layer id from layer name or '@'-qualified layer-id
def _layer_name_to_lid(net, layer_name: str):
    layer_lid = None
    if layer_name.startswith('@'):
        # This is a layer id
        lids = net.get_layers().keys()
        layer_lid = layer_name[1:]
        if not layer_lid in lids:
            raise ConversionError(f"Layer ID '{layer_lid}' not in: {list(lids)}")
            return None
        return layer_lid

    layer_names = [l.name for _, l in net.get_layers().items()]
    if net.get_org_platform() != 'onnx':
        layer = net.get_layer_by_name(layer_name)
        layer_lid = layer.lid if layer else None
    else:
        # In ONNX layer names are prefixed by their type followed by '_'. Remove this prefix
        for lid, l in net.get_layers().items():
            if l.name[l.name.find('_') + 1:] == layer_name:
                layer_lid = lid
                break
        else:
            layer_names = [name[name.find('_') + 1:] for name in layer_names]

    if not layer_lid:
        raise ConversionError(f"Layer name '{layer_name}' not in: {list(set(layer_names))}")
    return layer_lid


# Get lids of nodes in this branch from lid up to (but excluding) a node in the root_set.
# Return None if no root_set nodes reachable from lid
def _get_branch_nodes(net, lid, root_set: list, branch_nodes_cache:dict):
    if lid in root_set:
        return set()
    if lid in branch_nodes_cache:
        return branch_nodes_cache[lid]

    inputs = net.get_layer(lid).get_input_layers()
    parents_sets = [_get_branch_nodes(net, i_lid, root_set, branch_nodes_cache) for i_lid in inputs]
    if all(ps is None for ps in parents_sets):
        branch_nodes_cache[lid] = None
        return None
    parents = set([lid])
    for ps in parents_sets:
        if ps:
            parents = parents | ps
    branch_nodes_cache[lid] = parents
    return parents


# Get lids of nodes in the network that are after the root_id_set in execution order.
def _get_nodes_after(net, root_id_set: list):
    after_set = set()
    branch_nodes_cache = {}
    for out in net.get_output_layers():
        after_set_for_branch = _get_branch_nodes(net, out.lid, root_id_set, branch_nodes_cache)
        if after_set_for_branch:
            after_set = after_set | after_set_for_branch
    return list(after_set)


# Get data-type to be used for hybrid quantization.
def _hybrid_type(dtype: DataType):
    if dtype == DataType.FLOAT16:
        return 'float32'
    elif dtype == DataType.INT16:
        return 'dynamic_fixed_point-i16'
    raise ConversionError(f'invalid data_type specified for mixed quantization: {dtype.value}')


def run_command(command: str):
    # start the command with a shell, redirect stderr to stdout and pipe the stdout to us for reading
    logger.info(command)
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # log all lines received from the subprocess
    for line in p.stdout:
        logger.info(line.decode('utf-8', 'backslashreplace').strip())

    # wait for the subprocess to complete
    p.wait()

    # return the exit code of the process to the caller for further handling
    return p.returncode


def convert(subgraph: SerializedSubgraph, subgraph_meta: MetaInfo, delegate: DelegateInfo,
            conversion_options: ConversionOptions, work_dir: Path, out_dir: Path | None = None) -> ConvertedModel:
    """
    Converts a serialized subgraph to the EBG format suitable
    for execution on the NPU
    """

    # convert all paths to absolute because the converter code may change the cwd
    ebg_output_path = out_dir if out_dir is not None else work_dir
    model_path = str(subgraph.model_file.absolute())
    weights_path = str(subgraph.weights_file.absolute()) if subgraph.weights_file is not None else None
    out_dir = str(ebg_output_path.absolute())

    logger.debug("converting subgraph: %s", subgraph)
    logger.debug("meta: %s", subgraph_meta)

    # the converter code will mess around with files in the parent of the workdir so we create a subdir here
    # we also convert it to absolute path because the converter will do chdir
    converter_work_dir = (work_dir / "work").absolute()
    converter_work_dir.mkdir(parents=True)

    # save the cwd to ensure we switch back to it when we are done (the cwd may be changed by acuity)
    orig_cwd = os.getcwd()

    try:
        sc = Converter(conversion_options.verbose, conversion_options.silent, conversion_options.debug, str(converter_work_dir),
                str(conversion_options.cache_dir) if conversion_options.cache_dir else None)

        net = sc.convert(model_path, weights_path, subgraph_meta)

        input_count = sc.generate(net, subgraph_meta, conversion_options.target, out_dir,
                    str(conversion_options.tools_dir) if conversion_options.tools_dir else None,
                    str(conversion_options.vssdk_dir) if conversion_options.vssdk_dir else None,
                    True,
                    conversion_options.profiling, conversion_options.cpu_profiling)

    finally:
        # here we change back to the original cwd without if it is necessary because acuitylib may have
        # changed the current dir to a directory that was deleted in the meanwhile and os.getcwd will
        # fail with FileNotFoundError in that case
        os.chdir(orig_cwd)

    listing_files = [ebg_output_path / "model_info.txt"]
    if os.path.isfile(work_dir / "quantization_info.yaml"):
        listing_files += [work_dir / "quantization_info.yaml"]
    if os.path.isfile(work_dir / "quantization_entropy.txt"):
        listing_files += [work_dir / "quantization_entropy.txt"]

    return ConvertedModel(
        model_file=ebg_output_path / "model.nb",
        meta_file=ebg_output_path / "model.json",
        listing_files=listing_files,
        input_count=input_count,
        output_count=len(net.get_output_layers())
    )
