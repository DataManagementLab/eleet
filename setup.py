"""Install the package."""

import os
from setuptools import setup


def read(fname):
    """Read README file."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="eleet",
    version="0.0.1",
    author="Matthias.Urban",
    author_email="matthias.urban@cs.tu-darmstadt.de",
    description=("Multi-modal Database."),
    # license = "Apache",
    keywords="multi-modal database",
    packages=['eleet_pretrain', 'eleet'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        # "Topic :: Utilities",
        # "License :: OSI Approved :: Apache",
    ],
)
