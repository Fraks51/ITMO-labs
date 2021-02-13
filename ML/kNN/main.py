import math
import pandas as pd
import random


class Object:
    def __init__(self, x: [int], new_x: [int], distance_func_name: str, m: int):
        self.y = x[-1]
        self.distance = get_dist_func(distance_func_name, m)(x[0:-1], new_x[0:-1])
        self.one_hot = x[-3:]

    def __lt__(self, other):
        return self.distance < other.distance


def log(kernel_func: callable, dist_func: callable, window_type: str, window_parameter: float, F_measure: float):
    print(", ".join(["Kernel func = " + kernel_func, "Dist func = " + dist_func, "Window type = " + window_type,
                     "Window param = " + str(window_parameter), "F measure = " + str(F_measure)]))
    print("<" + "=" * 30 + ">")


def minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        if i == len(dataset[0]) - 1:
            continue
        value_min = dataset[:, i].min()
        value_max = dataset[:, i].max()
        minmax.append([value_min, value_max])
    return minmax


def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            if i == len(row) - 1:  # exclude labels
                continue
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    return dataset


def get_dist_func(func_name: str, m: int) -> callable:
    return {
        "manhattan": lambda x_1, x_2: sum([abs(x_1[i] - x_2[i]) for i in range(m)]),
        "euclidean": lambda x_1, x_2: sum([(x_1[i] - x_2[i]) ** 2 for i in range(m)]) ** 0.5,
        "chebyshev": lambda x_1, x_2: max([abs(x_1[i] - x_2[i]) for i in range(m)])
    }[func_name]


def get_kernel(func_name: str) -> callable:
    return {
        "uniform": lambda x: 0.5 if abs(x) < 1 else 0,
        "triangular": lambda x: 1 - abs(x) if abs(x) < 1 else 0,
        "epanechnikov": lambda x: 0.75 * (1 - x ** 2) if abs(x) < 1 else 0,
        "quartic": lambda x: (15 * (1 - x ** 2) ** 2) / 16 if abs(x) < 1 else 0,
        "triweight": lambda x: (35 * (1 - x ** 2) ** 3) / 32 if abs(x) < 1 else 0,
        "tricube": lambda x: (70 * (1 - abs(x) ** 3) ** 3) / 81 if abs(x) < 1 else 0,
        "gaussian": lambda x: math.exp(-0.5 * (x ** 2)) / ((2 * math.pi) ** 0.5),
        "cosine": lambda x: math.pi * math.cos(math.pi * x / 2) if abs(x) < 1 else 0,
        "logistic": lambda x: 1 / (math.exp(x) + 2 + math.exp(-x)),
        "sigmoid": lambda x: (2 / math.pi) * 1 / (math.exp(x) + math.exp(-x))
    }[func_name]


def get_macro_measure(CM, n) -> float:
    true_right = [CM[i][i] for i in range(n)]
    predicted_classes = [0 for i in range(n)]
    for i in range(n):
        for j in range(n):
            predicted_classes[i] += CM[j][i]
    classes = [sum(CM[i]) for i in range(n)]
    all_element = sum(classes)
    precision = sum([true_right[i] * classes[i] / predicted_classes[i] if predicted_classes[i] != 0 else 0.0
                     for i in range(n)]) / all_element
    recall = sum(true_right) / all_element
    return 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0.0


def parameterized_regression(kernel_func: str, dist_func: str, window_type: str, _window_parameter, is_one_hot,
                             n, m, data) -> float:
    len_data = len(data)
    confusion_matrix = [[0 for i in range(3)] for j in range(3)]
    K = get_kernel(kernel_func)
    for i in range(10):
        # cross-validation
        left_board = (len_data * i) // 10
        right_board = (len_data * (i + 1)) // 10
        matrix = data[0:left_board] + data[right_board:len_data]
        qs = data[left_board: right_board]
        for q in qs:
            objects = [Object(matrix[i], q, dist_func, m) for i in range(n)]
            if window_type != "fixed":
                objects.sort()
                window_parameter = objects[int(_window_parameter)].distance
            else:
                window_parameter = _window_parameter
            ws = []
            for obj in objects:
                if window_parameter != 0:
                    ws.append(K(obj.distance / window_parameter))
                elif obj.distance == 0:
                    ws.append(K(0))
                else:
                    ws.append(0)
            sum_ws = sum(ws)
            if is_one_hot:
                label_vector = []
                label = 0
                for j in range(3):
                    label_vector.append(sum([objects[i].one_hot[j] * ws[i] for i in range(n)]) / sum_ws if sum_ws != 0
                                        else (sum([obj.one_hot[j] for obj in objects]) / len(objects)))
                for j in range(3):
                    if label_vector[j] == max(label_vector):
                        label = j
                        break
                confusion_matrix[int(q[-4]) - 1][label] += 1
            else:
                label = sum([objects[i].y * ws[i] for i in range(n)]) / sum_ws if sum_ws != 0 else \
                    (sum([obj.y for obj in objects]) / len(objects))
                confusion_matrix[int(q[-1]) - 1][round(label) - 1] += 1
    return get_macro_measure(confusion_matrix, 3)


all_dist_func = ["manhattan", "euclidean", "chebyshev"]
all_kernel_func = ["uniform", "triangular", "epanechnikov", "quartic", "triweight", "tricube",
                   "gaussian", "cosine", "logistic", "sigmoid"]
all_window_type = {"variable": [2, 3, 5, 8, 10, 13], "fixed": [0.25, 0.5, 0.75, 1, 1.5]}


def get_normalized_dataset(filename):
    dataset_table = pd.read_csv(filename)
    min_max = minmax(dataset_table.values)
    normalized_dataset = normalize(dataset_table.values, min_max).tolist()
    random.shuffle(normalized_dataset)
    return normalized_dataset


if __name__ == '__main__':
    normalized_dataset_values = get_normalized_dataset('data/seeds.csv')
    len_dataset = len(normalized_dataset_values)
    n, m = len_dataset * 9 // 10, len(normalized_dataset_values[0]) - 1
    way = input("Simple or OneHot:")
    best_parameters = []
    best_F_measure = 0
    if way == "Simple":
        is_one_hot = False
    elif way == "OneHot":
        one_hot_matrix = [[1 if normalized_dataset_values[i][-1] == j + 1 else 0 for j in range(3)]
                          for i in range(len(normalized_dataset_values))]
        for i in range(len_dataset):
            normalized_dataset_values[i] += one_hot_matrix[i]
        is_one_hot = True
    else:
        print("Unknown way to ")
        exit(1)
    for _dist_func in all_dist_func:
        for _kernel_func in all_kernel_func:
            for _window_type in all_window_type.keys():
                for _window_size in all_window_type[_window_type]:
                    F_measure = parameterized_regression(_kernel_func, _dist_func, _window_type,
                                                         _window_size, is_one_hot, n, m, normalized_dataset_values)
                    log(_kernel_func, _dist_func, _window_type, _window_size, F_measure)
                    if F_measure > best_F_measure:
                        best_F_measure = F_measure
                        best_parameters = [_kernel_func, _dist_func, _window_type, _window_size]
    print("<|> Best parameters:")
    print(best_parameters)
