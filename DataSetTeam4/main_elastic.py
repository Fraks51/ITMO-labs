# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
import numpy as np
from numpy.linalg import linalg
from sklearn.linear_model import ElasticNet


def h_ws(ws: np.ndarray, _xs: np.ndarray) -> float:
    return np.sum(ws[0:-1] * _xs) + ws[-1]


def find_part_rmse(_xs: np.ndarray, _ys: np.ndarray, ws: np.ndarray) -> float:
    vector_mul = [h_ws(ws, _xs[i]) for i in range(len(_ys))]
    result = sum([(_ys[i] - vector_mul[i]) ** 2
                  for i in range(len(_ys))])
    return result


def mnk(xs: np.ndarray, ys: np.array, tau):
    n = len(xs)
    _xs = np.array([xs[i].tolist() + [1] for i in range(n)])
    ws = np.dot(np.dot(linalg.inv(np.dot(_xs.T, _xs) + tau * np.eye(len(_xs[1]))), _xs.T), ys)
    return ws


def get_x_ranking(x, y, num_of_columns):
    estimator = ElasticNet(alpha=0.05, l1_ratio=0.5, tol=0.0001)
    selector = RFE(estimator, n_features_to_select=num_of_columns, step=1)
    selector = selector.fit(x, y)
    support_map = dict(zip(x.columns, selector.support_))
    test_x = pd.read_csv("../input/csc-hw3-autumn2020-team-4/2020_hw3_team_4_ds_test.csv").replace({
                                                    "yes": 1, "no": 0,
                                                    "T": 0, "A": 1, "m": 1,
                                                    "f": 0, "o": 2, "M": 1, "F": 0})
    x = x.append(test_x)
    return x[filter(lambda feature: support_map[feature], x.columns)]


def get_rmse(new_x, y, tau):
    result = 0
    kf = KFold(n_splits=10)
    for train_indexes, test_indexes in kf.split(new_x, y):
        x_test, x_train = new_x[test_indexes], new_x[train_indexes]
        y_test, y_train = y[test_indexes], y[train_indexes]
        ws = mnk(x_train, y_train.to_numpy(), tau)
        result += find_part_rmse(x_test, y_test.to_numpy(), ws)
    return (result / len(y)) ** 0.5


def find_hyper():
    dataset = pd.read_csv("data/train4.csv")
    dataset = dataset.replace({"yes": 1, "no": 0, "T": 0, "A": 1, "m": 1, "f": 0, "o": 2, "M": 1, "F": 0})
    y = dataset["target"]
    x = dataset.drop(["target"], axis=1)
    for i in range(1, 30):
        print(i)
        new_x = get_x_ranking(x, y, i)
        scaler = MinMaxScaler()
        scaler.fit(new_x)
        new_x = scaler.transform(new_x)
        min_loss, min_tau = 1000, 0
        for t in range(1, 90300, 100): # range(30001, 30300, 10)
            tau = t / 100000
            loss = get_rmse(new_x, y, tau)
            if min_loss > loss:
                min_loss = loss
                min_tau = tau
            # print("Get for tau = " + str(tau))
            # print(get_rmse(new_x, y, tau))
        print(min_tau)
        print(min_loss)
        print("<+++++>")


def main():
    dataset = pd.read_csv("../input/csc-hw3-autumn2020-team-4/2020_hw3_team_4_ds_train_full.csv")
    dataset = dataset.replace({"yes": 1, "no": 0, "T": 0, "A": 1, "m": 1, "f": 0, "o": 2, "M": 1, "F": 0})
    y = dataset["target"]
    x = dataset.drop(["target"], axis=1)
    new_x = get_x_ranking(x, y, 9)
    scaler = MinMaxScaler()
    scaler.fit(new_x)
    new_x = scaler.transform(new_x)
    test_x = new_x[(len(y)):]
    train_x = new_x[:(len(y))]
    ws = mnk(train_x, y.to_numpy(), 0.369)
    ans = [h_ws(ws, i) for i in test_x]
    answer_df = pd.DataFrame()
    answer_df['id'] = pd.read_csv("../input/csc-hw3-autumn2020-team-4/2020_hw3_team_4_ds_test.csv")['id']
    answer_df['target'] = ans
    answer_df.to_csv("submisson.csv", index=False)


if __name__ == '__main__':
    main()
