[![PyPi Version](https://img.shields.io/pypi/v/compmec-nurbs.svg?style=flat-square)](https://pypi.org/project/compmec-nurbs/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/compmec-nurbs.svg?style=flat-square)](https://pypi.org/project/compmec-nurbs/)
![Tests](https://github.com/compmec/nurbs/actions/workflows/tests.yml/badge.svg)

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

Or install it manually

```
git clone https://github.com/compmec/nurbs
cd nurbs
pip install -e .
```

To verify if everything works in your machine, type the command in the main folder

```
pytest
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


[pypilink]: https://pypi.org/project/compmec-nurbs/
[issueslink]: https://github.com/compmec/nurbs/issues