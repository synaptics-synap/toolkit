from setuptools import setup, find_packages, Extension
import os, shutil, re
from pathlib import Path
from datetime import datetime

# Get the current pysynap version
def get_version():
    with open('pysynap/__init__.py', 'r') as init_file:
        init_py = init_file.read()
    version_match = re.search(r"^version = ['\"]([^'\"]*)['\"]", init_py, re.M)
    if version_match:
        version = version_match.group(1)
        snapshot = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{version}rc{snapshot}"
    raise RuntimeError("Unable to find version string.")

# Get the required dependecy list
def get_requirements(directory):
    requirements = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("requirements.txt"):
                with open(os.path.join(root, file), "r") as f:
                    for line in f:
                        line = line.strip()
                        # Skip lines starting with --extra-index-url
                        if not line.startswith('--extra-index-url'):
                            requirements.append(line)
    return requirements

requirements = get_requirements('.')

# Define the extension module
extensions = [
    Extension(
        name="pysynap.prebuilts",
        sources=[],
        libraries=["ebg_utils", "ovxlib"],
        library_dirs=["pysynap/prebuilts/lib/x86_64-linux-gcc"],
    ),
]

# Ensure the scripts are available in the package directory to be included in the wheel
os.makedirs('pysynap/scripts', exist_ok=True)
shutil.copyfile('synap_convert.py', 'pysynap/scripts/synap_convert.py')
shutil.copyfile('image_from_raw.py', 'pysynap/scripts/image_from_raw.py')
shutil.copyfile('image_to_raw.py', 'pysynap/scripts/image_to_raw.py')
shutil.copyfile('image_od.py', 'pysynap/scripts/image_od.py')

# Create synap_help.py script for handling help command
with open('pysynap/scripts/synap_help.py', 'w') as f:
    f.write("""\
import sys

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'help':
        print("SyNAP Toolkit\\n")
        print("Usage:\\n\\tCOMMAND ARGS\\n\\tRun 'COMMAND --help' for more information on a command.\\n")
        print("Commands:")
        print("  synap_convert        - Convert and compile model")
        print("  synap_image_from_raw - Convert image file to raw format")
        print("  synap_image_to_raw   - Generate image file from raw format")
        print("  synap_image_od       - Superimpose object-detection boxes to an image")
    else:
        print("Usage: synap help")

if __name__ == "__main__":
    main()
""")

setup(
    name='synap',
    version=get_version(),
    author='Synaptics',
    author_email='meetdineshbhai.patel@synaptics.com',
    description='Synaptics AI Toolkit',
    packages=find_packages(exclude=['**/.*', '**/.git']),
    python_requires='>=3.10, <3.13',
    install_requires=requirements,
    package_dir={
        'pysynap': 'pysynap'
    },
    package_data={
        'pysynap': ['*'],
    },
    include_package_data=True,
    ext_modules=extensions,
    classifiers=[ 'Operating System :: POSIX :: Linux','Programming Language :: Python :: 3.10','Programming Language :: Python :: 3.12',],
    entry_points={
        'console_scripts': [
            'synap_convert=pysynap.scripts.synap_convert:main',
            'synap_image_from_raw=pysynap.scripts.image_from_raw:main',
            'synap_image_to_raw=pysynap.scripts.image_to_raw:main',
            'synap_image_od=pysynap.scripts.image_od:main',
            'synap=pysynap.scripts.synap_help:main',
        ],
    },
)