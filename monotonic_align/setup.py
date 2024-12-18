import os
if not os.path.exists('monotonic_align'):
  os.mkdir('monotonic_align')

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'monotonic_align',
  ext_modules = cythonize("core.pyx"),
  include_dirs=[numpy.get_include()]
)
