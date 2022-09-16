#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
from setuptools import setup, find_packages
# from distutils.core import setup
import os
import stat
import shutil
import platform
import sys
import site
import glob


# -- file paths --
long_description="""Data Hub: A Central Location to Manage Data IO"""
setup(
    name='data_hub',
    version='100.100.100',
    description='A python Data Hub',
    long_description=long_description,
    url='https://github.com/gauenk/data_hub',
    author='Kent Gauen',
    author_email='gauenk@purdue.edu',
    license='MIT',
    keywords='datasets',
    install_requires=['easydict',"torch","pathlib"],
    packages=find_packages(),
)
