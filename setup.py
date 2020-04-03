import setuptools
from PhaseNet.version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PhaseNet",
    version=__version__,
    author="",
    author_email="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
    ],
    scripts=["PhaseNet/phasenet_run.py"],
    python_requires='>=3.6',
)
