[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Code Coverage][coverage-image]][coverage-url]
[![][versions-image]][versions-url]

# Nurbs

This repository contains code for inteporlate functions using B-Splines and Nurbs.


#### Features

* Basic Functions
    * Spline ```N```
    * Rational ```R```
    * Derivative
* Curves
    * Spline
    * Rational
* Knot operations
    * Insertion
    * Removal
* Degree operations
    * Degree elevation
    * Degree reduction

## Install

This library is available in [PyPI][pypilink]. To install it

```
pip install compmec-nurbs
```

## Documentation

In progress


# FAQ

#### Must I learn the theory to use it?

No! Just see the examples and it will be fine

#### Can I understand the code here?

Yes! The easier way is to look up the **python notebook** which contains the theory along examples

#### Is this code efficient?

No. It's written in python and the functions were made for easy understanding, not for performance.
That means: It's not very fast, but it works fine.


## Contribute

Please use the [Issues][issueslink] or refer to the email ```compmecgit@gmail.com```

<!-- Badges: -->

[pypi-image]: https://img.shields.io/pypi/v/compmec-nurbs
[pypi-url]: https://pypi.org/project/compmec-nurbs/
[build-image]: https://github.com/compmec/nurbs/actions/workflows/build.yaml/badge.svg
[build-url]: https://github.com/compmec/nurbs/actions/workflows/build.yaml
[coverage-image]: https://codecov.io/gh/compmec/nurbs/branch/main/graph/badge.svg
[coverage-url]: https://codecov.io/gh/compmec/nurbs/
[versions-image]: https://img.shields.io/pypi/pyversions/compmec-nurbs.svg?style=flat-square
[versions-url]: https://pypi.org/project/compmec-nurbs/
[pypilink]: https://pypi.org/project/compmec-nurbs/
[issueslink]: https://github.com/compmec/nurbs/issues
