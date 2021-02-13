import pandas as pd
import numpy as np
import sklearn
import random
import copy
import sys

eps = 0.001


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
    for j in range(len(dataset)):
        row = dataset[j]
        for i in range(len(row)):
            if i == len(row) - 1:  # exclude labels
                continue
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
        dataset[j] = np.array(row)
    return dataset


def rand_index(clusters, y):
    all = 0
    TP = 0
    FN = 0
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            all += 1
            if clusters[i] == clusters[j] and y[i] == y[j]:
                TP += 1
            if (not clusters[i] == clusters[j]) and (not y[i] == y[j]):
                FN += 1
    return (TP + FN) / all


def silhouette(clusters, X, clust_num):
    sum_ = 0
    arg_clust = [[] for _ in range(clust_num)]
    for i in range(len(X)):
        arg_clust[clusters[i]].append(X[i])
    for k in range(clust_num):
        for x in arg_clust[k]:
            a = (sum([dist(x, i) for i in arg_clust[k]])) / len(arg_clust[k]) \
                if len(arg_clust[k]) > 0 else 0
            b = min([sum([dist(i, x) for i in arg_clust[l]]) / len(arg_clust[l])
                     if not l == k and not len(arg_clust[l]) == 0
                     else sys.maxsize
                    for l in range(clust_num)])
            sum_ += (b - a) / max(a, b)
    return sum_ / len(X)


def diff(xs, ys):
    _diff = 0
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        for j in range(len(x)):
            _diff += abs(x[j] - y[j])
    return _diff


def dist(x, y):
    return sum([(x[i] - y[i]) ** 2 for i in range(len(x))])


def k_means(X, clusters_num):
    m = len(X[0])
    clusters = np.array([random.randint(0, clusters_num - 1) for _ in range(len(X))])
    centers = [np.sum([random.choice(X) for _ in range(5)], axis=0) / 5 for _ in range(clusters_num)]
    old_centers = [np.zeros(m) for _ in range(clusters_num)]
    while diff(centers, old_centers) > eps:
        args_clust = [[] for _ in range(clusters_num)]
        for i in range(len(X)):
            clusters_x = []
            for j in range(len(centers)):
                clusters_x.append(dist(X[i], centers[j]))
            clusters[i] = np.argmin(clusters_x)
            args_clust[clusters[i]].append(X[i])
        old_centers = copy.copy(centers)
        centers = [np.sum(np.array(args_clust[i]), axis=0) / len(args_clust[i])
                   if len(args_clust[i]) > 0
                   else np.zeros(m)
                   for i in range(clusters_num)]
        return clusters


def get_normalized_dataset(filename):
    dataset_table = pd.read_csv(filename)
    min_max = minmax(dataset_table.values)
    normalized_dataset = normalize(dataset_table.values, min_max).tolist()
    return np.array(normalized_dataset)


def main():
    normalized_dataset_values = get_normalized_dataset('data/seeds.csv')
    y = normalized_dataset_values[:, -1]
    normalized_dataset_values = normalized_dataset_values[:, :-1]
    clusters = k_means(normalized_dataset_values, 3)


if __name__ == '__main__':
    main()

    import main
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA


    def draw_plot(x, y, x1, y1):
        plt.plot(x, y, label='Rand index')
        plt.plot(x1, y1, label='Silhouette')
        plt.legend(loc='upper center')
        plt.show()


    normalized_dataset_values = main.get_normalized_dataset('../data/seeds.csv')
    y = normalized_dataset_values[:, -1]
    normalized_dataset_values = normalized_dataset_values[:, :-1]
    x = []
    rand = []
    silh = []

    for i in range(2, 15):
        clust = main.k_means(normalized_dataset_values, i)
        x.append(i)
        rand.append(main.rand_index(clust, y))
        silh.append(main.silhouette(clust, normalized_dataset_values, i))
    draw_plot(x, rand, x, silh)

import main
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def draw_plot(x, y, clazz):
    plt.scatter(x, y, c=clazz, cmap='winter', marker=(4,0))
    plt.figure(figsize=(12,12))
    plt.show()


normalized_dataset_values = main.get_normalized_dataset('../data/seeds.csv')
y = normalized_dataset_values[:, -1]
normalized_dataset_values = normalized_dataset_values[:, :-1]
clusters = main.k_means(normalized_dataset_values, 3)
pca = PCA(n_components=2)
new_xs = pca.fit_transform(normalized_dataset_values)
draw_plot(new_xs[:, 0], new_xs[:, 1], y)
draw_plot(new_xs[:, 0], new_xs[:, 1], clusters)
