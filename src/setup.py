from setuptools import setup
from setuptools import Extension

add_mat_module = Extension(name='base',
                           sources=['base.cpp'],
                           include_dirs=[r'/usr/include/',
                                         r'/usr/include/eigen3/',
                                         r'/home/usrname/anaconda3/lib/python3.9/site-packages/pybind11/include']
                           )

setup(ext_modules=[add_mat_module])


#python setup.py build_ext --inplace