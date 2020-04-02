import subprocess
from distutils.core import setup, Extension
# from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

ext_modules = [Extension("test_long", ["test_long.pyx"])]

setup(cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules, include_dirs=[np.get_include()])
subprocess.call(["cython", "-a", "test_long.pyx"])
# subprocess.call(["cython", "-a", "pure_cython.pyx"])
