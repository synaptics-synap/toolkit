#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

import argparse
import json
import shutil
import os
import struct
import subprocess
import collections
from io import BytesIO

from tempfile import TemporaryDirectory

USAGE = """

This tool encrypts and signs a SYNAP or EBG model.

It takes as input the model to be encrypted and a JSON file describing the security policy with the following format.

{ "inputs": [ X, Y, Z ], "outputs": [ V, W ] }

where X, Y, Z represent the inputs of the network (in this example the network has three inputs) and take values from:

 - "secure": the input must be secure memory
 - "non-secure": the input must be non secure memory
 - "any": the input can be either in secure or non-secure memory [DEFAULT]

and where V, W represent the outputs of the network (in this example the network has two outputs) and take values from:

 - "secure-if-input-secure": the output buffer must be secure if at least one input is in secure memory  [DEFAULT]
 - "any": the output can be either in secure or non-secure memory

If no security-policy file is specified, the default security will be used for all inpus and outputs.

"""


class Struct:
    def __init__(self, fields):
        self.fields = fields

    def format(self):
        return '<' + ''.join([x[1] for x in self.fields])

    def size(self):
        return struct.calcsize(self.format())

    def parse(self, data):
        fields_data = struct.unpack(self.format(), data[0:self.size()])
        return {y[0]: fields_data[x] for x, y in enumerate(self.fields)}, data[self.size():]

    def serialize(self, data):
        return struct.pack(self.format(), *[data[x[0]] for x in self.fields])


EXTRAS = Struct([('prod_image_flag', 'I'),
                 ('vendor_gid', 'I'),
                 ('cert_id', 'I'),
                 ('clear_payload_len', 'I')])


EBG_MEMORY_HEADER = Struct([('type', 'I'),
                            ('address', 'I'),
                            ('size', 'I'),
                            ('alignment', 'I'),
                            ('page_size', 'I')])

EBG_IO_HEADER = Struct([('address', 'I'),
                        ('size', 'I'),
                        ('alignment', 'I'),
                        ('page_size', 'I')])

EBG_METADATA_HEADER = Struct([('hardware_type', 'I'),
                              ('network_name', '64s'),
                              ('compiler_id', 'I'),
                              ('compiler_version', 'I'),
                              ('execution_mode', 'I'),
                              ('input_count', 'I'),
                              ('output_count', 'I'),
                              ('memory_area_count', 'I')])

EBG_IMAGE_HEADER = Struct([('magic', '4s'),
                           ('endianness', 'I'),
                           ('version', 'I'),
                           ('security_type', 'I'),
                           ('security_info_length', 'I'),
                           ('metadata_length', 'I'),
                           ('vsi_data_length', 'I'),
                           ('padding_length', 'I'),
                           ('code_length', 'I')])

EBG_AUTHENTICATED_PUBLIC_DATA_HEADER = Struct([('metadata_length', 'I'),
                                               ('vsi_data_length', 'I'),
                                               ('padding_length', 'I'),
                                               ('code_length', 'I'),
                                               ('input_count', 'I'),
                                               ('output_count', 'I')])

SECURE_IMAGE_HEADER = Struct([('image_type', 'I'),
                              ('image_format', 'I'),
                              ('length', 'I')])

ROOT_CERTIFICATE_HEADER = Struct([('iv', '16s'),
                                  ('checksum_clear', '32s'),
                                  ('segID', 'I'),
                                  ('segID_mask', 'I'),
                                  ('version', 'I'),
                                  ('version_mask', 'I'),
                                  ('production_image_flag', 'I'),
                                  ('ta_vendor_gid', 'I'),                             
                                  ('signature', '256s')])

MODEL_CERTIFICATE_HEADER = Struct([('iv', '16s'),
                                   ('checksum_clear', '32s'),
                                   ('segID', 'I'),
                                   ('segID_mask', 'I'),
                                   ('version', 'I'),
                                   ('version_mask', 'I'),
                                   ('production_image_flag', 'I'),
                                   ('ta_vendor_gid', 'I'),
                                   ('ta_cert_id', 'I'),
                                   ('signature', '256s')])

SECURE_MODEL_HEADER = Struct([('iv', '16s'),
                              ('checksum_clear', '32s'),
                              ('checksum_enc', '32s'),
                              ('segID', 'I'),
                              ('segID_mask', 'I'),
                              ('version', 'I'),
                              ('version_mask', 'I'),
                              ('production_image_flag', 'I'),
                              ('ta_vendor_gid', 'I'),
                              ('ta_cert_id', 'I'),
                              ('clear_payload_len', 'I'),
                              ('signature', '256s')])

# these values are used in the JSON file resp. the binary security policy to specific the policy
INPUT_TYPE_TO_ID = {'secure': 0, 'non-secure': 1, 'any': 2}
OUTPUT_TYPE_TO_ID = {'secure-if-input-secure': 0, 'any': 1}


def encrypt_model_code(enc_tool, clear_data, model_code, encryption_key, signature_key, model_cert_obj):

    if len(model_code) % 16 != 0:
        raise Exception("Code must be padded")

    prod_flag = model_cert_obj._cert_header['production_image_flag']
    vendor_gid = model_cert_obj._cert_header['ta_vendor_gid']
    cert_id = model_cert_obj._cert_header['ta_cert_id']
    seg_id = model_cert_obj._cert_header['segID']
    seg_id_mask = model_cert_obj._cert_header['segID_mask']
    version = model_cert_obj._cert_header['version']
    version_mask = model_cert_obj._cert_header['version_mask']

    with TemporaryDirectory() as tmp_dir:

        extras = EXTRAS.serialize({'prod_image_flag': prod_flag,
                                   'vendor_gid': vendor_gid,
                                   'cert_id': cert_id,
                                   'clear_payload_len': len(clear_data)})

        extras_file = os.path.join(tmp_dir, 'extras')

        with open(extras_file, 'wb') as fp:
            fp.write(extras)

        model_file = os.path.join(tmp_dir, 'network.ebg')

        with open(model_file, 'wb') as fp:
            fp.write(model_code)

        clear_data_file = os.path.join(tmp_dir, 'metadata.bin')

        with open(clear_data_file, 'wb') as fp:
            fp.write(clear_data)

        output_file = os.path.join(tmp_dir, 'encrypted.bin')

        command_line = [enc_tool,
                        '-t', 'MODEL',  # image type
                        '-k', encryption_key,  # encryption key
                        '-n', signature_key,  # signing key
                        '-s', hex(seg_id),  # segid
                        '-S', hex(seg_id_mask),  # segid mask
                        '-r', hex(version),  # version
                        '-R', hex(version_mask),  # version mask
                        '-x', extras_file,  # extras
                        '-l', '0x0',  # variable length payload
                        '-i', model_file,
                        '-I', clear_data_file,
                        '-o', output_file]

        # run the encryption tool
        print("Running encryption tool with command:")
        print(' '.join(command_line))

        try:
            subprocess.check_call(command_line)
        except subprocess.CalledProcessError as e:

            # ignore the error if larger than 0, this is a bug in the encryption tool
            # real error codes of the tool are always negative
            if e.returncode < 0:                    
                print(f"Error running encryption tool (exit code = {e.returncode})")
                exit(1)

        # read the output
        with open(output_file, 'rb') as fp:
            encrypted_model = fp.read()

        return encrypted_model


def split_secure_image(root_cert, model_cert, secure_image):
    """
    Retrieve from the secure image create by the external tool the security information data and the code section
    """

    image_hdr, payload = SECURE_IMAGE_HEADER.parse(secure_image)

    if image_hdr['image_type'] != 0x71:
        raise Exception

    model_header, payload = SECURE_MODEL_HEADER.parse(payload)

    auth_data, _ = EBG_AUTHENTICATED_PUBLIC_DATA_HEADER.parse(payload)

    code_offset = SECURE_IMAGE_HEADER.size() + SECURE_MODEL_HEADER.size() + model_header['clear_payload_len']

    security_info_length = SECURE_IMAGE_HEADER.size() + SECURE_MODEL_HEADER.size() + \
                           EBG_AUTHENTICATED_PUBLIC_DATA_HEADER.size() + \
                           (auth_data['input_count'] + auth_data['output_count'] ) * 4

    return root_cert + model_cert + secure_image[0:security_info_length], secure_image[code_offset:]


class RootCertificateImage:
    def __init__(self, cert):
        self._parse_header(cert)

    def _parse_header(self, cert_data):
        self._header, image_payload = SECURE_IMAGE_HEADER.parse(cert_data)

        if self._header['image_type'] != 0x19:
            raise Exception("Invalid image type 0x%02x (expected 0x19)" % self._header['image_type'])

        if self._header['image_format'] != 1:
            raise Exception("Unsupported image format %d" % self._header['image_format'])

        if self._header['length'] != len(cert_data):
            raise Exception("Invalid certificate size (expected %d found %d)" %
                            (self._header['length'], len(image_payload)))
        
        self._cert_header, cert_payload = ROOT_CERTIFICATE_HEADER.parse(image_payload)

        return cert_payload


class ModelCertificateImage:
    def __init__(self, cert):
        self._parse_header(cert)

    def _parse_header(self, cert_data):
        self._header, image_payload = SECURE_IMAGE_HEADER.parse(cert_data)

        if self._header['image_type'] != 0x1a:
            raise Exception("Invalid image type 0x%02x (expected 0x1a or 0x19)" % self._header['image_type'])

        if self._header['image_format'] != 1:
            raise Exception("Unsupported image format %d" % self._header['image_format'])

        if self._header['length'] != len(cert_data):
            raise Exception("Invalid certificate size (expected %d found %d)" %
                            (self._header['length'], len(image_payload)))
        
        self._cert_header, cert_payload = MODEL_CERTIFICATE_HEADER.parse(image_payload)

        return cert_payload


class ModelImage:

    def __init__(self, model):

        model_payload = self._parse_header(model)

        buf = BytesIO(model_payload)

        self._security_info = buf.read(self._header['security_info_length'])
        self._metadata = buf.read(self._header['metadata_length'])
        self._vsi_data = buf.read(self._header['vsi_data_length'])
        self._padding = buf.read(self._header['padding_length'])
        self._code = buf.read(self._header['code_length'])

        self._parse_metadata()

    def _parse_header(self, model_data):

        self._header, model_payload = EBG_IMAGE_HEADER.parse(model_data)

        if self._header['magic'] != b'EBGX':
            raise Exception("Invalid magic %s (expected EBGX)" % self._header['magic'])

        if self._header['endianness'] != 0:
            raise Exception("Big endian models are unsupported")

        if self._header['version'] != 1:
            raise Exception("Unsupported model version %d" % self._header['version'])

        model_payload_size = (self._header['code_length'] + self._header['metadata_length'] +
                              self._header['vsi_data_length'] + self._header['padding_length']
                              + self._header['security_info_length'])

        if model_payload_size != len(model_payload):
            raise Exception("Invalid model size (expected %d found %d)" % (model_payload_size, len(model_payload)))

        return model_payload

    def _parse_metadata(self):

        # parse the metadata header
        if len(self._metadata) < EBG_METADATA_HEADER.size():
            raise Exception("Malformed metadata length (expected at least %d, found %d)" %
                            (EBG_METADATA_HEADER.size(), self._header['metadata_length']))

        self._metadata_header, self._metadata_payload = EBG_METADATA_HEADER.parse(self._metadata)

        # parse the extra information about input/output and other memory areas
        memory_areas_size = (EBG_IO_HEADER.size() * (self._metadata_header['input_count'] +
                                                     self._metadata_header['output_count']) +
                             EBG_MEMORY_HEADER.size() * self._metadata_header['memory_area_count'])

        if len(self._metadata_payload) < memory_areas_size:
            raise Exception("Metadata doesn't include information about all memory areas")

    def serialize(self):
        return EBG_IMAGE_HEADER.serialize(self._header) + self._security_info + \
               self._metadata + self._vsi_data + self._padding + self._code

    def _generate_policy(self, security_config):

        if security_config:
            input_descs = security_config.get('inputs', [])
            output_descs = security_config.get('outputs', [])
    
            if self._metadata_header['input_count'] != len(input_descs):
                raise Exception("Received %d input security policies but the model has %d inputs" %
                                (len(input_descs), self._metadata_header['input_count']))
    
            if self._metadata_header['output_count'] != len(output_descs):
                raise Exception("Received %d output security policies but the model has %d output" %
                                (len(input_descs), self._metadata_header['input_count']))

        else:
            # Create default security policy
            input_descs = ['any'] * self._metadata_header['input_count']
            output_descs = ['secure-if-input-secure'] * self._metadata_header['output_count']

        for input_desc in input_descs:
            if input_desc not in INPUT_TYPE_TO_ID:
                print(f"Invalid input security policy {input_desc} (valid policies: {", ".join(INPUT_TYPE_TO_ID)})")
                exit(1)
            
        for output_desc in output_descs:
            if output_desc not in OUTPUT_TYPE_TO_ID:
                print(f"Invalid output security policy {output_desc} (valid policies: {", ".join(OUTPUT_TYPE_TO_ID)})")
                exit(1)

        print("Using security policy for inputs:")
        for idx, input_desc in enumerate(input_descs):
            print(f" {idx}: {input_desc}")
        print()

        print("Using security policy for outputs:")
        for idx, output_desc in enumerate(output_descs):
            print(f" {idx}: {output_desc}")
        print()

        return input_descs, output_descs

    def _create_public_data(self, input_descs, output_descs):
        security_config_out = BytesIO()

        # write out the public data header
        public_data = {'metadata_length': (EBG_METADATA_HEADER.size() + (self._metadata_header['input_count'] +
                                          self._metadata_header['output_count'] +
                                          self._metadata_header['memory_area_count']) *
                                          EBG_MEMORY_HEADER.size()),
                        'vsi_data_length': self._header['vsi_data_length'],
                        'padding_length': self._header['padding_length'],
                        'code_length': self._header['code_length'],
                        'input_count': len(input_descs),
                        'output_count': len(output_descs)}

        security_config_out.write(EBG_AUTHENTICATED_PUBLIC_DATA_HEADER.serialize(public_data))

        # write out the policy for the inputs
        for input_desc in input_descs:
            security_config_out.write(struct.pack('<I', INPUT_TYPE_TO_ID[input_desc]))

        # write out the policy for the outputs
        for output_desc in output_descs:
            security_config_out.write(struct.pack('<I', OUTPUT_TYPE_TO_ID[output_desc]))

        return security_config_out.getvalue()

    def encrypt_code(self, security_config, enc_tool, root_cert, model_cert, model_cert_obj,
                     encryption_key, signature_key):

        # check that the model is a clear model
        if self._header['security_type'] != 0:
            raise Exception("Cannot encrypt non-clear model")

        # strip vsi data and padding
        self._vsi_data = b''
        self._padding = b''
        self._header['padding_length'] = 0
        self._header['vsi_data_length'] = 0

        # pad the code if necessary
        padding_bytes = 16 - (len(self._code) % 16)

        if padding_bytes != 16:
            self._code += b'\0' * padding_bytes
            self._header['code_length'] = len(self._code)

        # create the authenticated public data and the encrypted code
        input_descs, output_descs = self._generate_policy(security_config)
        auth_public_data = self._create_public_data(input_descs, output_descs)
        clear_data = auth_public_data + self._metadata + self._vsi_data + self._padding

        # pad the clear data if necessary
        # enc_tool requires the clear data to be a multiple of 16 bytes
        padding_bytes = 16 - (len(clear_data) % 16)
        if padding_bytes != 16:
            self._header['padding_length'] = padding_bytes
            self._padding += b'\0' * padding_bytes
            auth_public_data = self._create_public_data(input_descs, output_descs)
            clear_data = auth_public_data + self._metadata + self._vsi_data + self._padding

        secure_image = encrypt_model_code(enc_tool, clear_data, self._code, encryption_key, signature_key, model_cert_obj)

        # update code and security info
        self._security_info, self._code = split_secure_image(root_cert, model_cert, secure_image)
        self._header['security_info_length'] = len(self._security_info)

        # set the model as encrypted
        self._header['security_type'] = 1


def encrypt_model_file(src_file, dst_file, security_policy, enc_tool, vendor_cert, model_cert, model_cert_obj, encryption_key, signature_key):
    # read the clear model
    with open(src_file, 'rb') as fp:
        model_data = fp.read()

    # parse the clear model
    model = ModelImage(model_data)

    # encrypt the model
    model.encrypt_code(security_policy, enc_tool, vendor_cert, model_cert, model_cert_obj, encryption_key, signature_key)

    # write out the encrypted model
    with open(dst_file, 'wb') as fp:
        fp.write(model.serialize())


def main():
    parser = argparse.ArgumentParser(description="Encrypts a SYNAP or EBG file")
    parser.add_argument('--enc-tool', required=True, help="Location of the image encryption tool")
    parser.add_argument('--security-policy', help="Optional JSON file with the input/output security policy")
    parser.add_argument('--vendor-certificate', required=True, help="Model vendor root certificate")
    parser.add_argument('--model-certificate', required=True, help="Model certificate")
    parser.add_argument('--encryption-key', required=True, help="Model encryption key (AES)")
    parser.add_argument('--signature-key', required=True, help="Model signature key (RSA)")
    parser.add_argument('model_file', help="Clear .synap or EBG file")
    parser.add_argument('output_file', help="Location of for the encrypted file")

    args = parser.parse_args()

    # read the security policy in JSON format if specified
    security_policy = None
    if args.security_policy:
        with open(args.security_policy) as fp:
            security_policy = json.load(fp)

    # read the root certificate for the model
    with open(args.vendor_certificate, 'rb') as fp:
        vendor_cert = fp.read()

    # read the model certificate
    with open(args.model_certificate, 'rb') as fp:
        model_cert = fp.read()

    # parse and print the vendor certificate information
    try:
        vendor_cert_obj = RootCertificateImage(vendor_cert)
    except Exception as e:
        print("Error parsing vendor certificate: %s" % str(e))
        exit(1)    

    print("Using root certificate with following information:")
    print("  Vendor GID: 0x%08x" % vendor_cert_obj._cert_header['ta_vendor_gid'])
    print("  Segmentation ID: 0x%08x" % vendor_cert_obj._cert_header['segID'])
    print("  Segmentation ID Mask: 0x%08x" % vendor_cert_obj._cert_header['segID_mask'])
    print("  Version: 0x%08x" % vendor_cert_obj._cert_header['version'])
    print("  Version Mask: 0x%08x" % vendor_cert_obj._cert_header['version_mask'])
    print("  Production Image Flag: %d" % vendor_cert_obj._cert_header['production_image_flag'])
    print("")

    # parse and print model certificate information
    try:
        model_cert_obj = ModelCertificateImage(model_cert)
    except Exception as e:
        print("Error parsing model certificate: %s" % str(e))
        exit(1)

    print("Using model certificate with following information:")
    print("  Vendor GID: 0x%08x" % model_cert_obj._cert_header['ta_vendor_gid'])
    print("  Certificate ID: 0x%08x" % model_cert_obj._cert_header['ta_cert_id'])
    print("  Segmentation ID: 0x%08x" % model_cert_obj._cert_header['segID'])
    print("  Segmentation ID Mask: 0x%08x" % model_cert_obj._cert_header['segID_mask'])
    print("  Version: 0x%08x" % model_cert_obj._cert_header['version'])
    print("  Version Mask: 0x%08x" % model_cert_obj._cert_header['version_mask'])
    print("  Production Image Flag: %d" % model_cert_obj._cert_header['production_image_flag'])    
    print("")

    if model_cert_obj._cert_header['ta_vendor_gid'] != vendor_cert_obj._cert_header['ta_vendor_gid']:
        print("Error: Model certificate vendor GID does not match the root certificate vendor GID")
        exit(1)

    if model_cert_obj._cert_header['production_image_flag'] != vendor_cert_obj._cert_header['production_image_flag']:
        print("Error: Model certificate production image flag does not match the root certificate production image flag")
        exit(1)
    
    # Check if the file is a SYNAP model
    _, file_extension = os.path.splitext(args.model_file)
    
    if file_extension.lower() == '.synap':
        print("Processing EBGs in SYNAP model...")
        print()
        
        # Extract zip file in a temporary directory
        with TemporaryDirectory() as tmp_dir:
            shutil.unpack_archive(args.model_file, tmp_dir, 'zip')            

            # Find the EBG files (.nb extension)
            for root, dirs, files in os.walk(tmp_dir):
                for file in files:
                    if file.endswith(".nb"):
                        print(f"Processing subgraph {os.path.relpath(root, tmp_dir)}/{file}...")
                        print()
                        ebg_file = os.path.join(root, file)
                        # Encrypt the EBG file
                        encrypt_model_file(ebg_file, ebg_file + ".enc",
                                           security_policy, args.enc_tool, vendor_cert, model_cert, model_cert_obj,
                                           args.encryption_key, args.signature_key)
                        # Move the encrypted EBG file to the original location
                        os.replace(ebg_file + ".enc", ebg_file)
                        # Update the security attributes in the corresponding model.json
                        input_descs = None
                        output_descs = None
                        if security_policy:
                            input_descs = security_policy.get('inputs', [])
                            output_descs = security_policy.get('outputs', [])
                        json_file = os.path.join(root, "model.json")
                        with open(json_file, 'r') as fp:
                            model_json = json.load(fp, object_pairs_hook=collections.OrderedDict)
                        model_json["secure"] = True
                        for i, input in enumerate(model_json["Inputs"].values()):
                            input["security"] = input_descs[i] if input_descs else "any"
                        for i, output in enumerate(model_json["Outputs"].values()):
                            output["security"] = output_descs[i] if output_descs else "secure-if-input-secure"
                        # save json file formatted
                        with open(json_file, 'w') as fp:
                            json.dump(model_json, fp, indent=4)

                        print("Subgraph encrypted")
                            
            
            # Zip the temporary directory
            with TemporaryDirectory() as tmp_out_dir:
                shutil.make_archive(tmp_out_dir + "/model.synap", 'zip', tmp_dir)
                shutil.copy(tmp_out_dir + "/model.synap.zip", args.output_file)

    else:
        # encrypt the .ebg model
        encrypt_model_file(args.model_file, args.output_file,
                           security_policy, args.enc_tool, vendor_cert, model_cert,
                           args.encryption_key, args.signature_key)


if __name__ == "__main__":
    main()
