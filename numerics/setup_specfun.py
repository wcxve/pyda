# # cython: language_level=3
# import sys
# import numpy
# import pybind11
# from distutils.core import setup, Extension
#
# source_files = ['setup_specfun.py']
# static_libraries = ['quadmath']
# static_lib_dir = '/opt/homebrew/opt/gcc/lib/gcc/current'
# libraries = ['quadmath']
# library_dirs = [
#     '/opt/homebrew/opt/gcc/lib/gcc/current',
#     '/Users/xuewc/heasoft-6.31.1/heacore/aarch64-apple-darwin22.3.0/lib',
#     '/opt/homebrew/opt/gcc/lib/gcc/current'
# ]
# include_dirs = [
#     numpy.get_include(),
#     pybind11.get_include(),
#     '/Users/xuewc/heasoft-6.31.1/heacore/aarch64-apple-darwin22.3.0/include'
# ]
#
# if sys.platform == 'win32':
#     libraries.extend(static_libraries)
#     library_dirs.append(static_lib_dir)
#     extra_objects = []
# else: # POSIX
#     extra_objects = ['{}/lib{}.dylib'.format(static_lib_dir, l) for l in static_libraries]
#
# ext = Extension(
#     'specfun',
#     sources=source_files,
#     libraries=libraries,
#     library_dirs=library_dirs,
#     include_dirs=include_dirs,
#     extra_compile_args=['-O3', '-fPIC', '-std=c++14', '-lm', '-lgsl', '-lquadmath'],
#     extra_objects=extra_objects,
#     language='c++',
# )
#
# setup(ext_modules=[ext])

import os
os.system('g++ specfun.cpp -Ofast -Wall -std=gnu++14 -fPIC `python3 -m pybind11 --includes` -I`pwd` -shared -o specfun`python3-config --extension-suffix` -undefined dynamic_lookup')