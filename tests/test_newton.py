import numpy as np
from numba import njit

import os; cwd = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.normpath(os.path.join(cwd, "..", ".."))
import sys; sys.path.append(package_path)

from calculation import optimizers


def compare_vectors(v1, v2, eps=1e-5):
    norm = np.linalg.norm(v1 - v2)
    assert norm < eps

def test_newton1():

    @njit
    def VF(X):
        return np.array([2 * X[0]**2 + X[1]**2 - 1, X[0]**3 + 6 * X[0]**2 * X[1] - 1])

    X0 = np.array([0.9, 0.18])
    print(f"newton started from {X0}")
    X = optimizers.newton(VF, X0)
    answer = np.array([0.68659442, 0.23911545])
    compare_vectors(answer, X)
    compare_vectors(VF(X), np.array([0, 0]))
    print(f"we got {X}, well done")

def test_newton2():

    @njit
    def VF(X):
        return np.array([2 * X[0]**2 + X[1]**2 - 1, X[0]**3 + 6 * X[0]**2 * X[1]])

    X0 = np.array([0.9, 0.18])
    print(f"newton started from {X0}")
    X = optimizers.newton(VF, X0, eps=1e-5)
    answer = np.array([0.70224689, -0.11704113])
    compare_vectors(answer, X)
    compare_vectors(VF(X), np.array([0, 0]))
    print(f"we got {X}, well done")


if __name__ == "__main__":
    test_newton1()
    test_newton2()
