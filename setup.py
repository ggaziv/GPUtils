import os
from setuptools import find_packages, setup

setup(
    name = "gputils",
    version = "0.1",
    author = "Guy Gaziv",
    description = "General Purpose Utilities",
    long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    packages=find_packages(),
)