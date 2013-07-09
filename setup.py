import sys

from distutils.core import setup
from distutils.extension import Extension


try:
    from Cython.Distutils import build_ext
except ImportError:
    sys.stderr.write("""
    ==================================================

    Please install Cython (http://cython.org/),
    which is required to build tree-hmm. Usually
    you can do:

        pip install -U cython

    or

        easy_install -U cython

    ==================================================
    """)
    sys.exit(1)

try:
    import numpy
except ImportError:
    sys.stderr.write("""
    ==================================================

    Please install numpy,
    which is required to build tree-hmm. Usually
    you can do:

        pip install -U numpy

    or

        easy_install -U numpy

    ==================================================
    """)
    sys.exit(1)


ext_modules = [Extension("treehmm.vb_mf", ["treehmm/vb_mf.pyx"],
                         #extra_compile_args=['-fopenmp', '-march=bdver1', '-mtune=bdver1', '-Ofast'],
                         #extra_link_args=['-fopenmp', '-march=bdver1', '-mtune=bdver1', '-Ofast'],
#                         extra_compile_args=['-fopenmp', '-I/data/apps/enthought_python/7.3.2/lib/python2.7/site-packages/numpy/core/include', '-L/data/apps/enthought_python/7.3.2/lib/'],
#                         extra_link_args=['-fopenmp', '-L/data/apps/enthought_python/7.3.2/lib/'],
                         include_dirs=[numpy.get_include()]),
               Extension("treehmm.vb_prodc", ["treehmm/vb_prodc.pyx"],
                         #extra_compile_args=['-fopenmp', '-march=bdver1', '-mtune=bdver1', '-Ofast'],
                         #extra_link_args=['-fopenmp', '-march=bdver1', '-mtune=bdver1', '-Ofast'],
#                         extra_compile_args=['-fopenmp', '-I/data/apps/enthought_python/7.3.2/lib/python2.7/site-packages/numpy/core/include', '-L/data/apps/enthought_python/7.3.2/lib/'],
#                         extra_link_args=['-fopenmp', '-L/data/apps/enthought_python/7.3.2/lib/'],
                         include_dirs=[numpy.get_include()]),
               Extension("treehmm.vb_prodc_sepTheta", ["treehmm/vb_prodc_sepTheta.pyx"],
                         #extra_compile_args=['-fopenmp', '-march=bdver1', '-mtune=bdver1', '-Ofast'],
                         #extra_link_args=['-fopenmp', '-march=bdver1', '-mtune=bdver1', '-Ofast'],
#                         extra_compile_args=['-fopenmp', '-I/data/apps/enthought_python/7.3.2/lib/python2.7/site-packages/numpy/core/include', '-L/data/apps/enthought_python/7.3.2/lib/'],
#                         extra_link_args=['-fopenmp', '-L/data/apps/enthought_python/7.3.2/lib/'],
                         include_dirs=[numpy.get_include()]),
               Extension("treehmm.clique_hmm", ["treehmm/vb_clique.pyx"],
                         #extra_compile_args=['-fopenmp', '-march=bdver1', '-mtune=bdver1', '-Ofast'],
                         #extra_link_args=['-fopenmp', '-march=bdver1', '-mtune=bdver1', '-Ofast'],
#                         extra_compile_args=['-fopenmp', '-I/data/apps/enthought_python/7.3.2/lib/python2.7/site-packages/numpy/core/include', '-L/data/apps/enthought_python/7.3.2/lib/'],
#                         extra_link_args=['-fopenmp', '-L/data/apps/enthought_python/7.3.2/lib/'],
                         include_dirs=[numpy.get_include()]),
                ]


install_requires = ['scipy', 'cython ( >= 0.15.1)']
if sys.version_info[:2] < (2, 7):
    install_requires += ['argparse']


setup(
  name = 'treehmm',
  description = 'Variational Inference for tree-structured Hidden-Markov Models',
  version = '0.1.0',
  author = 'Jake Biesinger, Yuanfeng Wang, Xiaohui Xie',
  author_email = 'jake.biesinger@gmail.com',
  url = 'https://github.com/uci-cbcl/tree-hmm',
  long_description = open('README.rst').read(),
  packages = ['treehmm', 'gmtkParam'],
  scripts = ['bin/tree-hmm'],
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
  requires = ['scipy', 'cython ( >= 0.15.1)', 'numpy'],
  classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: GNU General Public License (GPL)',
            'Topic :: Scientific/Engineering :: Bio-Informatics'],
)
