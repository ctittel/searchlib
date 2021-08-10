from setuptools import setup
from Cython.Build import cythonize

setup(
    setup_requires=['pbr'],
    pbr=True,
    ext_modules=cythonize('searchlib/*.pyx', language_level = "3", annotate=True)
)