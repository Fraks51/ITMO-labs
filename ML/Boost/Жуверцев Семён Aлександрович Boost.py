import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
import math
import random


def sign(x):
    if x < 0:
        return -1
    else:
        return 1


class AdaBoost:

    def __init__(self, steps):
        self.steps = steps
        self.algorithms = []
        self.algo_weights = []

    def fit(self, X, y):
        size = len(X)
        weights = [1 / size for _ in range(size)]
        for i in range(len(self.algorithms), self.steps):
            b = DecisionTreeClassifier(max_depth=2)
            b.fit(X, y, sample_weight=weights)
            predicted = b.predict(X)
            neg = sum([weights[j] if predicted[j] != y[j] else 0 for j in range(size)])
            algo_weight = (math.log((1 - neg) / neg)) / 2 if neg != 0 else 1 / 2
            for j in range(size):
                weights[j] = weights[j] * math.exp(-algo_weight * y[j] * predicted[j])
            w_sum = sum(weights)
            for j in range(size):
                weights[j] = weights[j] / w_sum
            self.algorithms.append(b)
            self.algo_weights.append(algo_weight)

    def predict(self, X):
        y = []
        for i in range(len(X)):
            y.append(sign(sum([self.algo_weights[j] * self.algorithms[j].predict([X[i]])[0]
                               for j in range(self.steps)])))
        return y


def get_normalized_data(name: str):
    data = pd.read_csv(name)
    data = data.replace({"P": 1, "N": -1})
    y = data["class"]
    X = data[["x", "y"]]
    return y, X


def get_accuracy(y, X, steps, kfold):
    accuracy = 0
    for train_index, test_index in kfold:
        X_train, X_test = X.iloc[train_index, :].values, X.iloc[test_index, :].values
        y_train, y_test = y[train_index].values, y[test_index].values
        boost = AdaBoost(steps)
        boost.fit(X_train, y_train)
        predicted = boost.predict(X_test)
        accuracy += sum([1 if predicted[i] == y_test[i] else 0
                         for i in range(len(y_test))])
    accuracy = accuracy / len(y)
    return accuracy


def main():
    y, X = get_normalized_data("data/chips.csv")
    # y = y.values
    # X = X.values
    kf = KFold(n_splits=5, shuffle=True)
    split = list(kf.split(X))
    for i in [1, 2, 3, 5, 8, 13, 21, 34, 55]:
        # boost = AdaBoost(i)
        # boost.fit(X, y)
        # predicted = boost.predict(X)
        # print(sum([1 if predicted[j] == y[j] else 0
        #            for j in range(len(y))]) / len(y))
        print(get_accuracy(y, X, i, split))


if __name__ == '__main__':
    main()
#%%

import main
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold


def draw_plot(x, y):
    plt.plot(x, y)
    plt.ylabel('Accuracy')
    plt.xlabel('Steps')
    plt.show()

#%%

y, X = main.get_normalized_data("../data/chips.csv")
x_0 = []
y_0 = []
kf = KFold(n_splits=5, shuffle=True)
split = list(kf.split(X))
for i in [1, 2, 3, 5, 8, 13, 21, 34, 55]:
    x_0.append(i)
    y_0.append(main.get_accuracy(y, X, i, split))
draw_plot(x_0, y_0)

#%%



#%%
#%%

import main
import numpy as np
import matplotlib.pyplot as plt


def predict(xx, yy, f):
    xx_ = xx.ravel()
    yy_ = yy.ravel()
    z = []
    for i in range(len(xx_)):
        f_res = f(xx_[i], yy_[i])
        z.append(main.sign(f_res))
    z = np.array(z)
    return z.reshape(xx.shape)


def draw_plot(x, y, clazz, boost, steps):
    x_ = np.arange(-1.5, 1.5, 0.01)
    y_ = np.arange(-1, 1, 0.01)
    xx, yy = np.meshgrid(x_, y_)
    f = lambda x1, x2: boost.predict([[x1, x2]])[0]
    z = predict(xx, yy, f)
    plt.contourf(xx, yy, z, cmap='GnBu')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1, 1)
    plt.scatter(x, y, c=clazz, cmap='winter', marker=(4,0), label='steps {}'.format(steps))
    plt.legend(loc='upper center')
    plt.figure(figsize=(12,12))
    plt.show()

#%%

y, X = main.get_normalized_data("../data/chips.csv")
for i in [1, 2, 3, 5, 8, 13, 21, 34, 55]:
    boost = main.AdaBoost(i)
    boost.fit(X.values, y.values)
    draw_plot(X['x'].values.tolist(), X['y'].values.tolist(), y.values.tolist(), boost, i)
