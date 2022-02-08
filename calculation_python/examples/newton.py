import os
import numpy as np
from numba import njit

cwd = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.normpath(os.path.join(cwd, "..", ".."))
import sys; sys.path.append(package_path)

# Every import of our library should looks like this
from calculation_python import optimizers

def main():

    @njit
    def VF(X):
        return np.array([2 * X[0]**2 + X[1]**2 - 1, X[0]**3 + 6 * X[0]**2 * X[1] - 1])

    X0 = np.array([0.9, 0.18])
    print(f"we start from {X0}")
    X = optimizers.newton(VF, X0)
    print(f"we got from {X}")

if __name__ == "__main__":
    main()
