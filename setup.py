from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("vb_mf", ["vb_mf.pyx"],
                         #extra_compile_args=['-fopenmp', '-march=bdver1', '-mtune=bdver1', '-Ofast'],
                         #extra_link_args=['-fopenmp', '-march=bdver1', '-mtune=bdver1', '-Ofast'],
                         extra_compile_args=['-fopenmp', '-I/data/apps/enthought_python/7.3.2/lib/python2.7/site-packages/numpy/core/include', '-L/data/apps/enthought_python/7.3.2/lib/'],
                         extra_link_args=['-fopenmp', '-L/data/apps/enthought_python/7.3.2/lib/'],
                         include_path=[numpy.get_include()]),
               Extension("vb_prodc", ["vb_prodc.pyx"],
                         #extra_compile_args=['-fopenmp', '-march=bdver1', '-mtune=bdver1', '-Ofast'],
                         #extra_link_args=['-fopenmp', '-march=bdver1', '-mtune=bdver1', '-Ofast'],
                         extra_compile_args=['-fopenmp', '-I/data/apps/enthought_python/7.3.2/lib/python2.7/site-packages/numpy/core/include', '-L/data/apps/enthought_python/7.3.2/lib/'],
                         extra_link_args=['-fopenmp', '-L/data/apps/enthought_python/7.3.2/lib/'],
                         include_path=[numpy.get_include()]),
               Extension("vb_prodc_sepTheta", ["vb_prodc_sepTheta.pyx"],
                         #extra_compile_args=['-fopenmp', '-march=bdver1', '-mtune=bdver1', '-Ofast'],
                         #extra_link_args=['-fopenmp', '-march=bdver1', '-mtune=bdver1', '-Ofast'],
                         extra_compile_args=['-fopenmp', '-I/data/apps/enthought_python/7.3.2/lib/python2.7/site-packages/numpy/core/include', '-L/data/apps/enthought_python/7.3.2/lib/'],
                         extra_link_args=['-fopenmp', '-L/data/apps/enthought_python/7.3.2/lib/'],
                         include_path=[numpy.get_include()]),
               Extension("clique_hmm", ["vb_clique.pyx"],
                         #extra_compile_args=['-fopenmp', '-march=bdver1', '-mtune=bdver1', '-Ofast'],
                         #extra_link_args=['-fopenmp', '-march=bdver1', '-mtune=bdver1', '-Ofast'],
                         extra_compile_args=['-fopenmp', '-I/data/apps/enthought_python/7.3.2/lib/python2.7/site-packages/numpy/core/include', '-L/data/apps/enthought_python/7.3.2/lib/'],
                         extra_link_args=['-fopenmp', '-L/data/apps/enthought_python/7.3.2/lib/'],
                         include_path=[numpy.get_include()]),
                        ]
setup(
  name = 'Variational Bayes Inference for lineage HMMs',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
