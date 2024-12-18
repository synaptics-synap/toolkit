# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

import sys

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'help':
        print("SyNAP Toolkit")
        
        print("Usage:\n\tCOMMAND ARGS\n\tRun 'COMMAND --help' for more information on a command.")
        
        print("Commands:")
        print("  synap_convert        - Convert and compile model")
        print("  synap_image_from_raw - Convert image file to raw format")
        print("  synap_image_to_raw   - Generate image file from raw format")
        print("  synap_image_od       - Superimpose object-detection boxes to an image")
        print("  synap_export_yolo    - Export YOLOv8 and YOLOv9 models to synap")
        print("  synap_export_clean   - Cleanup model export files")
        print("  synap_copy           - Copy exported synap models to board")
        print("  synap_profile        - Profile exported models on board with synap_cli")
    else:

        print("Usage: synap help")

if __name__ == "__main__":
    main()
