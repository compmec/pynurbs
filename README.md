[![Documentation Status][docs-img]][docs-url]
[![Build Status][build-img]][build-url]
[![Lint with Black][lintblack-img]][lintblack-url]
[![Code Coverage][coverage-img]][coverage-url]

[![PyPI Version][pypi-img]][pypi-url]
[![Python Versions][pyversions-img]][pyversions-url]
[![License: MIT][license-img]][license-url]

![compmec-nurbs logo](docs/source/img/logo.svg)

A object-oriented python package for parametrized geometry which supports [Custom objects](https://compmec-nurbs.readthedocs.io/en/latest/rst/custom_objects.html).

For now, it supports only 1D-objects (curves).

## Features

* [X] Evaluate points
* [X] Insert and remove knots
* [X] Degree increase and decrease
* [X] Split and unite curves
* [X] Math operations (``+``, ``-``, ``*``, ``/``, ``@``)
* [X] Projection of point in curve
* [X] Intersection of two curves
* [X] Derivative of curves
* [X] Line Integral
* [X] Curve fitting
* [X] Function fitting
* [X] Points fitting
* [ ] Reparameterize curve

## Install

This library is available in [PyPI][pypi-url]. To install it

```
pip install compmec-nurbs
```

For more details, refer to the [documentation][docs-url].

## Documentation

The documentation can be found at [compmec-nurbs.readthedocs.io][docs-url]


## Contribute

Please use the [Issues][issues-url] or refer to the email ```compmecgit@gmail.com```

<!-- Badges: -->

<!-- Badges: -->

[nurbswiki-url]: https://pt.wikipedia.org/wiki/NURBS
[lintblack-img]: https://github.com/compmec/nurbs/actions/workflows/black.yaml/badge.svg
[lintblack-url]: https://github.com/compmec/nurbs/actions/workflows/black.yaml
[docs-img]: https://readthedocs.org/projects/compmec-nurbs/badge/?version=latest
[docs-url]: https://compmec-nurbs.readthedocs.io/en/latest/?badge=latest
[pypi-img]: https://img.shields.io/pypi/v/compmec-nurbs
[pypi-url]: https://pypi.org/project/compmec-nurbs/
[build-img]: https://github.com/compmec/nurbs/actions/workflows/build.yaml/badge.svg
[build-url]: https://github.com/compmec/nurbs/actions/workflows/build.yaml
[coverage-img]: https://codecov.io/gh/compmec/nurbs/branch/main/graph/badge.svg?token=vfGMPe9W3I
[coverage-url]: https://codecov.io/gh/compmec/nurbs
[pyversions-img]: https://img.shields.io/pypi/pyversions/compmec-nurbs.svg?style=flat-square
[pyversions-url]: https://pypi.org/project/compmec-nurbs/
[license-img]: https://img.shields.io/pypi/l/ansicolortags.svg
[license-url]: https://github.com/compmec/nurbs/blob/main/LICENSE.md
[pypi-url]: https://pypi.org/project/compmec-nurbs/
[issues-url]: https://github.com/compmec/nurbs/issues
