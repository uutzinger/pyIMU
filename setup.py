#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This is a python install script written for pyIMU python package.
# pip3 install --upgrade setuptools wheel Cython build
#
# py -3 setup.py build_ext --inplace
# py -3 setup.py bdist
# py -3 setup.py sdist
# py -3 setup.py bdist_wheel
# pip3 install -e . # -e makes symlinks to the source folder and allows to edit the source code without having to reinstall the package

import io
import os
from setuptools import setup, find_packages
# from distutils.core import setup
# from Cython.Build import cythonize

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Get the list of Cython module files in the subfolder
cython_modules_path = "pyIMU"
module_files = [os.path.join(cython_modules_path, f) for f in os.listdir(cython_modules_path) if f.endswith(".py")]
excluded_files = [os.path.join(cython_modules_path, "__init__.py") ]

setup(
    name='pyIMU',
    version='1.0.0',
    description=("Python implementation of AHRS with motion." ),
    url='https://github.com/uutzinger/pyIMU',
    author='Urs Utzinger',
    author_email='uutzinger@gmail.com',
    # ext_modules = cythonize(module_files, exclude=excluded_files, compiler_directives=dict(language_level = "3")),
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    keywords='IMU, AHRS, Madgwick, quaternion, vector, motion, pose',
    # packages=["pyIMU", "pyIMU.quaternion", "pyIMU.utilities", "pyIMU.madgwick", "pyIMU.motion"],
    packages = find_packages(),
    install_requires=['numpy>1.0'],
    classifiers=[
        # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: IMU :: AHRS :: Sensor',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ]
)
