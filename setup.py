import os
import re
import sys

from setuptools import find_packages
from setuptools import setup

package_name = "engine"
exclusions = ["docs", "examples"]
packages = find_packages(exclude=exclusions)

with open(os.path.join(package_name, "__init__.py"), "rt") as fh:
    _version = re.search(
        '__version__\s*=\s*"(?P<version>.*)"\n',
        fh.read()
    ).group("version")

_install_requires = [
    "numpy",
    "numpyro",
    "jax",
    "jaxlib",
    "pyro-ppl",
    "torch"
]

_parameters = {
    "install_requires": _install_requires,
    "name": package_name,
    "packages": packages,
    "platform": "any",
    "url": "https://github.com/LFITaskForce/engine",
    "version": _version}

setup(**_parameters)
