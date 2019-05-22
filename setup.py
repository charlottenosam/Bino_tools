#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup

setup(
    name="Binotools",
    version='0.1.0',
    author="Charlotte Mason",
    author_email="charlote.mason@cfa.harvard.edu",
    packages=["binotools"],
    license="LICENSE",
    description="Tools for dealing with astronomical spectral energy distributions",
    long_description=open("README.md").read(),
    # package_data={"sedpy": ["data/*fits", "data/filters/*par"]},
    # include_package_data=True,
    install_requires=["astropy", "lmfit"],
)
