# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import math
import random
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold


class Object:
    def __init__(self, x: [int], new_x: [int], distance_func_name: str, m: int, y):
        self.y = x[-1]
        self.distance = get_dist_func(distance_func_name, m)(x, new_x)
        self.one_hot = y

    def __lt__(self, other):
        return self.distance < other.distance


def log(kernel_func: callable, dist_func: callable, window_type: str, window_parameter: float, F_measure: float):
    print(", ".join(["Kernel func = " + kernel_func, "Dist func = " + dist_func, "Window type = " + window_type,
                     "Window param = " + str(window_parameter), "F measure = " + str(F_measure)]))
    print("<" + "=" * 30 + ">")


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


def prepare_values(df, column):
    parsed_values = map(
        lambda x: x.replace('[', '').replace(']', '').split(','),
        df[column].to_numpy()
    )
    all_classes = set()
    parsed_values_copy = []
    for i in parsed_values:
        parsed_values_copy.append(i)
        for j in i:
            if j != "None" and str(j) != 'nan':
                all_classes.add(j)
    df = df.drop([column], axis=1)
    for _class in all_classes:
        class_res = []
        for k in parsed_values_copy:
            if _class in k:
                class_res.append(1)
            else:
                class_res.append(0)
        df[column + _class] = class_res
    return df


def parameterized_regression(kernel_func: str, dist_func: str, window_type: str, _window_parameter,
                             n, m, x, y, x_test):
    K = get_kernel(kernel_func)
    matrix = x
    qs = x_test
    ans = []
    con = 1
    for q in qs:
        con += 1
        objects = [Object(matrix[i], q, dist_func, m, y[i]) for i in range(n)]
        if window_type != "fixed":
            objects.sort()
            window_parameter = objects[int(_window_parameter)].distance
        else:
            window_parameter = _window_parameter
        ws = []
        if window_type == "fixed":
            for obj in objects:
                if window_parameter != 0:
                    ws.append(K(obj.distance / window_parameter))
                elif obj.distance == 0:
                    ws.append(K(0))
                else:
                    ws.append(0)
        else:
            for i in range(_window_parameter):
                obj = objects[i]
                if window_parameter != 0:
                    ws.append(K(obj.distance / window_parameter))
                elif obj.distance == 0:
                    ws.append(K(0))
                else:
                    ws.append(0)
        sum_ws = sum(ws)
        label_vector = []
        label = 0
        if window_type == "fixed":
            for j in range(7):
                label_vector.append(sum([objects[i].one_hot[j] * ws[i] for i in range(n)]) / sum_ws if sum_ws != 0
                                    else (sum([obj.one_hot[j] for obj in objects]) / len(objects)))
        else:
            for j in range(7):
                label_vector.append(
                    sum([objects[i].one_hot[j] * ws[i] for i in range(_window_parameter)]) / sum_ws if sum_ws != 0
                    else objects[0].one_hot[j])
        for j in range(7):
            if label_vector[j] == max(label_vector):
                label = j
                break
        ans.append(label + 3)
    return ans


def get_part_rmse(pred, y_test):
    ans = []
    for i in range(len(y_test)):
        label = 0
        for j in range(7):
            if y_test[i][j] == max(y_test[i]):
                label = j
                break
        ans.append(label + 3)
    return sum([1 if pred[i] == ans[i] else 0 for i in range(len(y_test))])


def get_rmse(kernel, dist_func, window, h, x, y, m):
    result = 0
    kf = KFold(n_splits=15)
    for train_indexes, test_indexes in kf.split(x, y, ):
        x_test, x_train = x[test_indexes], x[train_indexes]
        y_test, y_train = y[test_indexes], y[train_indexes]
        pred = parameterized_regression(kernel, dist_func, window, h, len(x_train), m, x_train, y_train, x_test)
        result += get_part_rmse(pred, y_test)
    return result / len(x)


all_dist_func = ["manhattan", "euclidean", "chebyshev"]
all_kernel_func = ["uniform", "triangular", "epanechnikov", "quartic", "triweight", "tricube",
                   "gaussian", "cosine", "logistic", "sigmoid"]
all_window_type = {"variable": [2, 3, 5, 8, 10, 13], "fixed": [0.25, 0.5, 0.75, 1, 1.5]}


def find_hyper():
    dataset = pd.read_csv("data/train7.csv").head(1000).replace({"w": 1, "r": 0})
    dataset = pd.get_dummies(dataset, columns=['target'])
    y = dataset[["target_3", "target_4", "target_5", "target_6", "target_7", "target_8", "target_9"]].values
    colums = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12"]
    x = dataset[colums]  # genre characters
    scaler = MinMaxScaler()
    scaler.fit(x)
    new_x = scaler.transform(x)
    min_loss, min_tau = 1, 0
    params = []
    counter = 1
    for kernel in all_kernel_func:
        for dist_f in all_dist_func:
            for window in all_window_type:
                for h in all_window_type[window]:
                    loss = get_rmse(kernel, dist_f, window, h, new_x, y, len(colums)) # def get_rmse(kernel, dist_func, window, h, x, y, m):
                    if min_loss < loss:
                        min_loss = loss
                        params = [kernel, dist_f, window]
                    print("kernel: {}, dist: {}, window: {}, h: {}".format(kernel, dist_f, window, h))
                    print("Loss: {}".format(loss))
    print("Best loss")
    print(min_loss)
    print("tau: {}, alpha: {}, l1: {}".format(params[0], params[1], params[2]))



def main():
    dataset = pd.read_csv("../input/csc-hw3-autumn2020-team-3/2020_hw3_team_3_ds_train_full.csv").replace({"w": 1, "r": 0})
    dataset = pd.get_dummies(dataset, columns=['target'])
    y = dataset[["target_3", "target_4", "target_5", "target_6", "target_7", "target_8", "target_9"]].values
    colums = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12"]
    dataset = dataset.append(pd.read_csv("../input/csc-hw3-autumn2020-team-3/2020_hw3_team_3_ds_test.csv"))
    x = dataset[colums]  # genre characters
    scaler = MinMaxScaler()
    scaler.fit(x)
    new_x = scaler.transform(x)
    x_test, x_train = new_x[len(y):], new_x[0:len(y)]
    pred = parameterized_regression("uniform", "manhattan", "variable", 20, len(x_train), len(colums), x_train, y, x_test)
    answer_df = pd.DataFrame()
    answer_df['id'] = pd.read_csv("../input/csc-hw3-autumn2020-team-3/2020_hw3_team_3_ds_test.csv")['id']
    answer_df['target'] = pred
    answer_df.to_csv("submisson.csv", index=False)


def cor_martix():
    df = pd.read_csv("data/train3.csv")
    corr = df.corr()
    corr.style.background_gradient(cmap='coolwarm')


if __name__ == '__main__':
    find_hyper()