from distutils.core import setup
from Cython.Build import cythonize


setup(name='recognition_programme',ext_modules=cythonize("recognition_picture.pyx"))



