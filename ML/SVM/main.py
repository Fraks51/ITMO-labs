import random
import copy
import pandas as pd
import numpy as np
import math


def log(kernel_func: callable, kernel_const: float, C: float, accuracy: float):
    print(", ".join(["Kernel func = " + kernel_func, "const for kernel = " + str(kernel_const),
                     "C = " + str(C), "Accuracy = " + str(accuracy)]))
    print("<" + "=" * 30 + ">")


def sign(x: float):
    return 1 if x >= 0 else -1


def get_weights_final(K, m: int, C: float, max_iterations: int, x, y):
    return support_vector_machine(K, m, C, max_iterations, x, y, 0.1)


def get_accuracy(K, m: int, C: float, max_iterations: int, x, y) -> float:
    accuracy = 0
    for i in range(4):
        left_board = (m * i) // 4
        right_board = (m * (i + 1)) // 4
        new_x = x[0:left_board] + x[right_board:]
        new_y = y[0:left_board] + y[right_board:]
        alphas, b = support_vector_machine(K, len(new_x), C, max_iterations, new_x, new_y, 0.1)
        new_x = np.array(new_x)
        for j in range(left_board, right_board):
            if y[j] == sign(sum([alphas[k] * new_y[k] * K(x[j], new_x[k])
                                 for k in range(len(new_x))]) + b):
                accuracy += 1
    return accuracy / m


def support_vector_machine(K, m: int, C: float, max_iterations: int,
                           x: [[float]], y: [float], t_c):
    lambdas = [0.0 for _ in range(m)]
    K_dist = [[0.0 for _ in range(m)] for _ in range(m)]
    for i in range(m):
        for j in range(i, m):
            K_dist[i][j] = K(x[i], x[j])
            K_dist[j][i] = K_dist[i][j]
    b = 0
    global_iteration = 0
    iteration, eps = 0, 0.00001
    while iteration < max_iterations and global_iteration < 2 * 10 ^ 17:
        was_change_alphas = False
        for i in range(m):
            loss_func_i = sum([lambdas[k] * y[k] * K_dist[i][k] for k in range(m)]) + b - y[i]
            if (y[i] * loss_func_i < -t_c and lambdas[i] < C - eps) \
                    or (y[i] * loss_func_i > t_c and lambdas[i] > eps):
                j = random.randint(0, m - 1)
                while j == i:
                    j = random.randint(0, m - 1)
                loss_func_j = (sum([lambdas[k] * y[k] * K_dist[k][j]
                                    for k in range(m)]) + b) - y[j]
                lam_i_old, lam_j_old = copy.copy(lambdas[i]), copy.copy(lambdas[j])
                if y[i] != y[j]:
                    l, h = max(0.0, lambdas[j] - lambdas[i]), min(C, C + lambdas[j] - lambdas[i])
                else:
                    l, h = max(0.0, lambdas[i] + lambdas[j] - C), min(C, lambdas[i] + lambdas[j])
                if abs(l - h) < eps:
                    continue
                k_i_j, k_i_i, k_j_j = K_dist[i][j], K_dist[i][i], K_dist[j][j]
                diff_xs = 2 * k_i_j - k_i_i - k_j_j
                if diff_xs > -eps:
                    continue
                lambdas[j] = lambdas[j] - y[j] * (loss_func_i - loss_func_j) / diff_xs
                lambdas[j] = h if lambdas[j] > h else (l if lambdas[j] < l else lambdas[j])
                if abs(lambdas[j] - lam_j_old) < eps:
                    continue
                lambdas[i] += y[i] * y[j] * (lam_j_old - lambdas[j])
                b1 = b - loss_func_i - y[i] * (lambdas[i] - lam_i_old) * k_i_i \
                     - y[j] * (lambdas[j] - lam_j_old) * k_i_j
                b2 = b - loss_func_j - y[i] * (lambdas[i] - lam_i_old) * k_i_j \
                     - y[j] * (lambdas[j] - lam_j_old) * k_j_j
                if 0 < lambdas[i] < C:
                    b = b1
                elif 0 < lambdas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                was_change_alphas = True
        if was_change_alphas:
            iteration = 0
        else:
            iteration += 1
        global_iteration += 1
    return lambdas, b


def get_kernel(kernel_name):
    return {
        "linear_kernel": lambda x, x_: sum([x[i] * x_[i] for i in range(len(x))]),
        "polynomial_kernel": lambda d, x, x_: (sum([x[i] * x_[i] for i in range(len(x))])) ** d,
        "gauss_kernel": lambda beta, x, x_: math.exp(-beta * sum([(x[i] - x_[i]) ** 2 for i in range(len(x))]))
    }[kernel_name]


kernels = {"linear_kernel": [0], "polynomial_kernel": [2, 3, 4, 5], "gauss_kernel": [1, 2, 3, 4, 5]}
all_C = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]


def main():
    dataset = pd.read_csv("data/chips.csv")
    data_size = len(dataset)
    data_values = dataset.replace('P', 1).replace('N', -1).values.tolist()
    random.shuffle(data_values)
    x, y = [], []
    for i in data_values:
        y.append(i[2])
        x.append(i[0:2])
    best_score = -10
    best_score_params = []
    for kernel_func in kernels:
        for kernel_const in kernels[kernel_func]:
            for C in all_C:
                if kernel_func == "linear_kernel":
                    accuracy = get_accuracy(get_kernel(kernel_func), data_size, C, 2, x, y)
                else:
                    K = lambda _x, _y: get_kernel(kernel_func)(kernel_const, _x, _y)
                    accuracy = get_accuracy(K, data_size, C, 2, x, y)
                if accuracy > best_score:
                    best_score = accuracy
                    best_score_params = [kernel_func, str(kernel_const), str(C)]
                log(kernel_func, kernel_const, C, accuracy)
    print("Best Score: " + str(best_score))
    print(", ".join(best_score_params))


if __name__ == '__main__':
    main()
