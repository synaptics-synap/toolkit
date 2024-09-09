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
    else:

        print("Usage: synap help")

if __name__ == "__main__":
    main()
