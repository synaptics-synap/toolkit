from setuptools import setup, find_packages, Extension
import os, shutil, re
from pathlib import Path
from datetime import datetime

# Get the current date and time as version
def get_version():
    return datetime.now().strftime("%Y%m%d%H%M%S")

# Get the required dependecy list
def get_requirements(directory):
    requirements = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("requirements.txt"):
                with open(os.path.join(root, file), "r") as f:
                    requirements.extend(f.read().splitlines())
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

os.makedirs('pysynap/scripts', exist_ok=True)
shutil.copyfile('synap_convert.py', 'pysynap/scripts/synap_convert.py')

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.rst").read_text()

setup(
    name='synap',
    version=get_version(),
    author='Synaptics',
    author_email='meetdineshbhai.patel@synaptics.com',
    description='Synaptics AI Toolkit',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=['**/.*', '**/.git']),
    python_requires='==3.10.*',
    install_requires=requirements,
    package_dir={
        'pysynap': 'pysynap'
    },
    package_data={
        'pysynap': ['*'],
    },
    include_package_data=True,
    ext_modules=extensions,
    classifiers=[ 'Operating System :: POSIX :: Linux','Programming Language :: Python :: 3.10',],
    entry_points={
        'console_scripts': [
            'synap_convert=pysynap.scripts.synap_convert:main',
        ],
    },
)