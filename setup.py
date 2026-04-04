from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("_k_means_merge.pyx")
)
