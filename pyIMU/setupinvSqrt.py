#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# py -3 setupinvSqrt.py build_ext --inplace

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("invSqrt.pyx")
)
