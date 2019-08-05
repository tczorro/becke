#!/usr/bin/env python
from setuptools import find_namespace_packages, setup
from Cython.Build import cythonize
import numpy

setup(
    name="becke",
    version='0.1.0',
    description='Fast and memory efficient implimentation of becke weights.',
    auther='Derrick Yang',
    author_email='yxt1991@gmail.com',
    url='https://github.com/tczorro/becke.git',
    package_dir={"": "src"},
    ext_modules=cythonize("src/becke/*.pyx"),
    include_dirs=[numpy.get_include()],
    packages=find_namespace_packages(where="src"),
    zip_safe=False,
    install_requires=["numpy>=1.16",]
)
