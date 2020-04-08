# run with python3 setup.py build_ext --inplace

import subprocess
from distutils.core import setup, Extension
# from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

ext_modules = [Extension("int_cy", ["int_cy.pyx"])]

setup(cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules, include_dirs=[np.get_include()])
subprocess.call(["cython", "-a", "int_cy.pyx"])
# subprocess.call(["cython", "-a", "pure_cython.pyx"])
