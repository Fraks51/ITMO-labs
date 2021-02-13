import random
import numpy as np
from numpy.linalg import linalg

eps = 0.001
learn_rate = 0.01
lambda_const = 0.01
maximum_iterations = 2000
tau = 0.0005


def mnk(xs: np.ndarray, ys: np.array):
    _xs = np.array([xs[i].tolist() + [1] for i in range(n)])
    ws = np.dot(np.dot(linalg.inv(np.dot(_xs.T, _xs) + tau * np.eye(n + 1)), _xs.T), ys)
    return ws


def h_ws(ws: np.ndarray, _xs: np.ndarray) -> float:
    return np.sum(ws[0:-1] * _xs) + ws[-1]


def gradient_step(ws: np.ndarray, _xs: np.ndarray, _ys: np.ndarray, learning_rate: float) -> np.ndarray:
    new_ws = np.zeros(len(ws), dtype=float)
    for i in range(len(_xs)):
        new_ws = new_ws + ((h_ws(ws, _xs[i]) - _ys[i]) * np.concatenate((_xs[i], [1]), axis=0))
    return ws * (1 - learning_rate * tau) + (-learning_rate / len(_xs)) * new_ws


def find_loss(_xs: np.ndarray, _ys: np.ndarray, ws: np.ndarray) -> float:
    result = sum([(h_ws(ws, _xs[i]) - _ys[i]) ** 2 for i in range(len(_ys))])
    return result / (len(_ys) * 2) + np.sum(ws ** 2) * tau


def find_loss_smape(_xs: np.ndarray, _ys: np.ndarray, ws: np.ndarray) -> float:
    vector_mul = [h_ws(ws, _xs[i]) for i in range(len(_ys))]
    result = sum([abs(vector_mul[i] - _ys[i]) / (abs(_ys[i]) + abs(vector_mul[i]))
                  for i in range(len(_ys))])
    return result / (len(_ys))


def normalise(m: int, xs: np.ndarray):
    for row in xs:
        for i in range(m):
            row[i] = (row[i] - mins[i]) / (maxs[i] - mins[i]) if maxs[i] != mins[i] else 1


def denormalise(m: int, ws: np.ndarray) -> np.ndarray:
    for i in range(m):
        if maxs[i] != mins[i]:
            k = ws[i] / (maxs[i] - mins[i])
            ws[i] = k
            ws[-1] -= mins[i] * k
        else:
            ws[i] = ws[i] / maxs[i]
    return ws


def start_gradient():
    global xs, ys
    ws = np.array([random.uniform(-1 / (2 * n), 1 / (2 * n)) for i in range(m + 1)], dtype=float)
    Q = -100
    all_indexes = [i for i in range(n)]
    new_Q = find_loss(xs, ys, ws)
    iteration = 0
    pointer = n
    while not (abs(new_Q - Q) < eps) and iteration < maximum_iterations:
        Q = new_Q
        if pointer >= n:
            random.shuffle(all_indexes)
            pointer = 0
        _xs, _ys = [], []
        for i in range(pointer, min(pointer + 45, n)):
            _xs.append(xs[all_indexes[i]])
            _ys.append(ys[all_indexes[i]])
        _xs, _ys = np.array(_xs, dtype=float), np.array(_ys, dtype=float)
        pointer += min(pointer + 45, n)
        ws = gradient_step(ws, _xs, _ys, learn_rate)
        eps_i = find_loss(_xs, _ys, ws)
        new_Q = (1 - lambda_const) * Q + lambda_const * eps_i
        iteration += 1
    return ws


def read_data(data_size):
    _xs = []
    _ys = []
    for i in range(data_size):
        input_numbers = [int(i) for i in f.readline().split()]
        for j in range(m):
            mins[j] = min(mins[j], input_numbers[j])
            maxs[j] = max(maxs[j], input_numbers[j])
        _xs.append(input_numbers[:-1])
        _ys.append(input_numbers[-1])
    _xs, _ys = np.array(_xs, dtype=float), np.array(_ys, dtype=float)
    return mins, maxs, _xs, _ys


if __name__ == '__main__':
    f = open("3.txt")
    m = int(f.readline())
    n = int(f.readline())
    mins = np.array([200000000000 for i in range(m)], dtype=float)
    maxs = np.array([-200000000000 for i in range(m)], dtype=float)
    mins, maxs, xs, ys = read_data(n)
    n_test = int(f.readline())
    mins, maxs, xs_test, ys_test = read_data(n_test)
    normalise(m, xs)
    normalise(m, xs_test)
    gradient_ws = start_gradient()
    print("Gradient result:")
    print(gradient_ws)
    print(find_loss_smape(xs_test, ys_test, gradient_ws))
    mnk_ws = mnk(xs, ys)
    print("Mnk result:")
    print(mnk_ws)
    print(find_loss_smape(xs_test, ys_test, mnk_ws))

