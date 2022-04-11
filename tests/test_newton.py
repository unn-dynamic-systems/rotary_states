import numpy as np

from unn_ds import optimizers

def VF1(X):
    return np.array([2 * X[0] ** 2 + X[1] ** 2 - 1, X[0] ** 3 + 6 * X[0] ** 2 * X[1] - 1])

def VF2(X):
    return np.array([2 * X[0] ** 2 + X[1] ** 2 - 1, X[0] ** 3 + 6 * X[0] ** 2 * X[1]])

def compare_vectors(v1, v2, eps=1e-5):
    norm = np.linalg.norm(v1 - v2)
    assert norm < eps

def test_newton1():
    X0 = np.array([0.9, 0.18])
    print(f"newton started from {X0}")
    X = optimizers.newton(VF1, X0)
    answer = np.array([0.68659442, 0.23911545])
    compare_vectors(answer, X)
    compare_vectors(VF1(X), np.array([0, 0]))
    print(f"we got {X}, well done")


def test_newton1_yacoby():
    
    def yacoby_matrix(X):
        return np.array([[4 * X[0], 2 * X[1]],
                        [3 * X[0] ** 2 + 12 * X[0] * X[1], 6 * X[0] ** 2]])

    X0 = np.array([0.9, 0.18])
    print(f"newton started from {X0}")
    X = optimizers.newton(VF1, X0, yacoby_matrix=yacoby_matrix)
    answer = np.array([0.68659442, 0.23911545])
    compare_vectors(answer, X)
    compare_vectors(VF1(X), np.array([0, 0]))
    print(f"we got {X}, well done")

def test_newton3():

    X0 = np.array([0.9, 0.18])
    print(f"newton started from {X0}")
    X = optimizers.newton(VF2, X0, eps=1e-5)
    answer = np.array([0.70224689, -0.11704113])
    compare_vectors(answer, X)
    compare_vectors(VF2(X), np.array([0, 0]))
    print(f"we got {X}, well done")


if __name__ == "__main__":
    test_newton1()
    test_newton1_yacoby()
    test_newton3()
