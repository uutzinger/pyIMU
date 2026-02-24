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
from setuptools import setup, find_packages, Extension
import numpy as np

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

def build_extensions():
    """
    Build optional Cython extensions when Cython is available.
    If Cython is not installed, fall back to pre-generated C sources when present.
    """
    module_names = ["_qcore", "_vcore", "_mcore", "_motion_core"]

    def extension_for(module_name: str, use_pyx: bool):
        suffix = ".pyx" if use_pyx else ".c"
        source_path = os.path.join("pyIMU", f"{module_name}{suffix}")
        return Extension(
            name=f"pyIMU.{module_name}",
            sources=[source_path],
            include_dirs=[np.get_include()],
        )

    try:
        from Cython.Build import cythonize
        cython_available = True
    except Exception:
        cython_available = False

    extensions = []
    if cython_available:
        pyx_extensions = []
        c_extensions = []
        for module_name in module_names:
            pyx_path = os.path.join(here, "pyIMU", f"{module_name}.pyx")
            if os.path.exists(pyx_path):
                pyx_extensions.append(extension_for(module_name, use_pyx=True))
            else:
                c_path = os.path.join(here, "pyIMU", f"{module_name}.c")
                if os.path.exists(c_path):
                    c_extensions.append(extension_for(module_name, use_pyx=False))
        extensions.extend(c_extensions)
        if pyx_extensions:
            extensions.extend(
                cythonize(
                    pyx_extensions,
                    compiler_directives={"language_level": "3"},
                    annotate=False,
                )
            )
        return extensions

    for module_name in module_names:
        c_path = os.path.join(here, "pyIMU", f"{module_name}.c")
        if os.path.exists(c_path):
            extensions.append(extension_for(module_name, use_pyx=False))

    return extensions

setup(
    name='pyIMU',
    version='1.0.0',
    description=("Python implementation of AHRS with motion." ),
    url='https://github.com/uutzinger/pyIMU',
    author='Urs Utzinger',
    author_email='uutzinger@gmail.com',
    ext_modules=build_extensions(),
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
