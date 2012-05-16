from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("vb_mf", ["vb_mf.pyx"],
                         #extra_compile_args=['-fopenmp', '-march=bdver1', '-mtune=bdver1', '-Ofast'],
                         #extra_link_args=['-fopenmp', '-march=bdver1', '-mtune=bdver1', '-Ofast'],
                         extra_compile_args=['-fopenmp'],
                         extra_link_args=['-fopenmp'],
                         include_path=[numpy.get_include()]),
               Extension("vb_prodc", ["vb_prodc.pyx"],
                         #extra_compile_args=['-fopenmp', '-march=bdver1', '-mtune=bdver1', '-Ofast'],
                         #extra_link_args=['-fopenmp', '-march=bdver1', '-mtune=bdver1', '-Ofast'],
                         extra_compile_args=['-fopenmp'],
                         extra_link_args=['-fopenmp'],
                         include_path=[numpy.get_include()]),
               #Extension("loopy_bp", ["loopy_bp_cython.pyx"],
               #          extra_compile_args=['-fopenmp'],
               #          extra_link_args=['-fopenmp'],
               #          include_path=[numpy.get_include()])
                        ]
setup(
  name = 'Variational Bayes Inference for lineage HMMs',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
