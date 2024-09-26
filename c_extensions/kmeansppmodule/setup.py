from setuptools import setup, Extension

module = Extension("mykmeanssp", sources=["kmeansppmodule.c"])
setup(
    name="mykmeanssp",
    version="1.0",
    description="python wraper for kmean++",
    ext_modules=[module],
)
