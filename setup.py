"""Domenico_Asimomitis_ICML_2023 setup.py."""

import json
import subprocess
from os.path import abspath, dirname, join

from setuptools import find_packages, setup

ROOT = abspath(dirname(__file__))

# please put setup keywords in the setup.json to keep this file clean
with open(join(ROOT, "setup.json"), "r") as f:
    SETUP = json.load(f)

# see 4 > https://packaging.python.org/guides/single-sourcing-package-version/
with open(join(ROOT, "VERSION"), "r") as f:
    VERSION = f.read().strip()

setup(
    # single source package version
    version=VERSION,
    # in combination with recursive-includes in MANIFEST.in, non-python files
    # within the click_egreport will be copied into the
    # site-packages and wheels installation directories
    include_package_data=True,
    # return a list all Python packages found within the ROOT directory
    packages=find_packages(),
    # pass parameters loaded from setup.json including author and version
    **SETUP,
)
