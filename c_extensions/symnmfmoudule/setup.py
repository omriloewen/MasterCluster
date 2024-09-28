from setuptools import setup, Extension

module = Extension("mysymnmf", sources=["symnmfmodule.c"])
setup(
    name="mysymnmf",
    version="1.0",
    description="python wrapper for symnmf",
    ext_modules=[module],
)
