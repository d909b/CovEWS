"""
Copyright (C) 2020  Patrick Schwab, F. Hoffmann-La Roche Ltd
Copyright (C) 2019  Patrick Schwab, ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
from distutils.core import setup
from setuptools import find_packages

setup(
    name='covews',
    version='1.0.0',
    packages=find_packages(),
    url='schwabpatrick.com',
    author='Patrick Schwab, Arash Mehrjou, Stefan Bauer',
    author_email='patrick.schwab@roche.com',
    license="MIT License",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "six >= 1.11.0",
        "scikit-learn == 0.22.2",
        "sklearn-pandas >= 1.8.0",
        "numpy >= 1.14.5",
        "scipy",
        "pandas>=1.0.3",
        "torch>=1.5.0",
        "torchtuples>=0.2.0",
        "lifelines>=0.24.8",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
