#!/usr/bin/env python
from setuptools import find_namespace_packages, setup
from Cython.Build import cythonize
import numpy

setup(
    name="becke",
    package_dir={"": "src"},
    ext_modules=cythonize("src/becke/*.pyx"),
    include_dirs=[numpy.get_include()],
    packages=find_namespace_packages(where="src"),
    zip_safe=False,
    install_requires=["numpy>=1.16",]
)
