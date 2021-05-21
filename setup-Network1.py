from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "Apply-Network1",
        ["Apply-Network1.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='Apply-Network1',
    ext_modules=cythonize(ext_modules),
)
