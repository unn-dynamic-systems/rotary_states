[![Pytest](https://github.com/unn-dynamic-systems/calculation/actions/workflows/python-tests.yml/badge.svg?branch=main)](https://github.com/unn-dynamic-systems/calculation/actions/workflows/python-tests.yml)

# Calculation
This repository is dedicated to numerical calculation of ODE's

## IDE setup
We use [vscode](https://code.visualstudio.com/) with a few plugins:
* [Git grapg](https://marketplace.visualstudio.com/items?itemName=mhutchie.git-graph) and [Git lens](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens) for better experience with git


## Setup environment and Test
```bash
pip3 install poetry && poetry install && poetry run pytest
```
---
**NOTE**

We use [poetry](https://python-poetry.org/) to manage dependencies.
