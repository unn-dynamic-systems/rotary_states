[![Pytest](https://github.com/unn-dynamic-systems/calculation/actions/workflows/python-tests.yml/badge.svg?branch=main)](https://github.com/unn-dynamic-systems/calculation/actions/workflows/python-tests.yml)

# Rotary states
*The minimalistic framework for finding rotational regimes and determining their stability*

Consider a system of autonomous ODE that describes $N$ coupled phase oscillators:

$$
\dot{X} = F(X), \ X \in R^n
$$

where $X$ is state vector of coupled phase oscillators. 
Without losing generality, we can write $X$ as follows:

$$
X = (\varphi_1, \dot{\varphi}_1, \varphi_2, \dot{\varphi}_2, ... , \varphi_N, \dot{\varphi}_N).
$$

This framework allows you to find rotational regimes and determine their stability.
## Rotational regimes finding

The rotation modes are described by the **rotation period**, the **phase period** and the **initial conditions**.

* **rotation period** $\in R$
* **phase period** $= 2\pi \cdot k, \ k \in Z$
* **initial conditions** $\in R^n$

If you want to find some rotational mode, you need to have a few things:
* Approximate rotation period: $T_0 \in R$
* Approximate initial conditions:

$$
IC_0 = (\varphi_{1_0}, \dot{\varphi}_{1_0}, \varphi_{2_0}, \dot{\varphi}_{2_0}, ... , \varphi_{N_0}, \dot{\varphi}_{N_0})
$$

* Phase period: $phase\\_period = 2\pi \cdot k$

> **Without losing generality for approximate initial conditions, we can set the $\varphi_{1_0} = 0$ and this is required for $IC_0$**


Code snippet for finding $4\pi$ rotary states
in system $\dot{X} = F(X)$:
```python
from rotary_states import limit_cycles
# define 'IC_0', 'T_0', 'F', 'args'
assert IC_0[0] == 0

# 'IC_0' - Approximate initial conditions
# 'T_0' - Approximate rotation period
# 'F' - Right side of system
# 'args' - A tuple of constants used in 'F'
# 'IC' - Initial conditions we found
# 'T' - Rotation period we found

T, IC = limit_cycles.find_limit_cycle(F, args, IC_0, T_0, phase_period=4*mt.pi)
```

For example, we will consider a chain of identic coupled oscillators with inertia $m$, friction
$\lambda$ and constant rotational
moment $\gamma$:

$$
\ddot{\varphi}_i + \lambda \dot{\varphi}_i + \sin{\varphi}_i = \gamma + k \left[ \sin(\varphi_{i+1} - \varphi_i) + \sin(\varphi_{i-1} - \varphi_i) \right].
$$

There are a lot of some rotational regimes you can find in that system. Also you can see the our previous [publication](http://doi.org/10.1063/5.0044521) about this study, but in this [example](./examples/limit_cycle.py) you can see how we can find some $4\pi$-periodic rotational regime with specific parameters in that system.

## Determination of the stability of the rotational mode
To determine the stability of the rotational mode we can use the [Floquet theory](https://www.wikiwand.com/en/Floquet_theory).

First of all we have to linearize the system around rotational regime we interested in by replacement:

$$
\varphi_i = \delta\varphi_i + \psi_i,
$$

we get the following system:

$$
\dot{\delta\varphi} = A(t)\delta\varphi,
$$

where $A(t) = A(t + T)$ - periodic matrix.

That system describes the perturbations. Stability of zero solution $\delta \varphi_i$ determine stability of rotational regime
$\psi_i$.

The eigenvalues of Monodromy matrix are determine zero solution stability of $\delta\varphi$.

If we have **rotation period**, **phase period** and **initial conditions** of the rotational regime we are interested in, we can determine their stability a that way:

```python
from rotary_states import limit_cycles
from numpy.linalg import eig
# 'IC' - Initial conditions
# 'T' - Rotation period
# 'F' - Right side of system
# 'F_linear' - A right side of system linearized around the rotational regime
# 'args' - A tuple of constants used in 'F'
# 'args_linear' - A tuple of constants used in 'F_linear'
# 'args' - A tuple of constants used in 'F'
# 'M' - Monodromy matrix we found.

M = limit_cycles.get_monogrommy_matrix(F, F_linear, IC, T, args_linear, args)

# Eigenvalues
eigenvalues, _ = eig(M)

# Stability
if (np.absolute(eigenvalues) < 1).all():
    print("STABLE")
else:
    print("UNSTABLE")
```
See the example [here](./examples/limit_cycle.py)

## Setup environment
> We use [poetry](https://python-poetry.org/) to manage dependencies

Poetry install:
```bash
pip3 install poetry
```

Install dependencies
```bash
# in root of this dir
poetry install
```
## Run tests

```bash
# in root of this dir
poetry run pytest
```
