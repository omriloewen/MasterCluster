from setuptools import setup, Extension

module = Extension("mysymnmf", sources=["c_extensions/symnmfmoudule/symnmfmodule.c"])
setup(
    name="mysymnmf",
    version="1.0",
    description="python wrapper for symnmf",
    ext_modules=[module],
)
