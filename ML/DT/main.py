import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import random
import numpy as np


def split_data(data):
    y = data["y"].values
    x = data.drop(["y"], axis=1).values
    return x, y


def get_accuracy(dataset, test_data, max_depth, criterion, split):
    x, y = split_data(dataset)
    decisionTree = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, splitter=split)
    decisionTree.fit(x, y)
    x_test, y_test = split_data(test_data)
    n = len(x_test)
    correct_answers = 0
    predict_y = decisionTree.predict(x_test)
    for i in range(n):
        if predict_y[i] == y_test[i]:
            correct_answers += 1
    return correct_answers / n


def find_min_and_max():
    min_depth, min_depth_data_set_num = None, None
    max_depth, max_depth_data_set_num = None, None
    for i in range(1, 22):
        num = str(i).rjust(2, '0')
        best_params = []
        best_accur = 0
        dataset = pd.read_csv("data/" + num + "_train.csv")
        test_data = pd.read_csv("data/" + num + "_test.csv")
        for criterion in ["gini", "entropy"]:
            for split in ["best", "random"]:
                for depth in range(1, 100):
                    acc = get_accuracy(dataset, test_data, depth, criterion, split)
                    if acc > best_accur:
                        best_params = [depth, criterion, split]
                        best_accur = acc
        if min_depth is None or min_depth > best_params[0]:
            min_depth = best_params[0]
            max_depth_data_set_num = i
        if max_depth is None or max_depth < best_params[0]:
            max_depth = best_params[0]
            max_depth_data_set_num = i
        print("Best params for " + num + " dataset:")
        print(best_params)
        print("Accuracy: " + str(best_accur))
        print("<" + "=" * 32 + ">")
    print("Max depth = " + str(max_depth))
    print("Dataset: " + str(max_depth_data_set_num))
    print("Min depth = " + str(min_depth))
    print("Dataset: " + str(min_depth_data_set_num))


def make_set_with_repetitions(x, y):
    y_values = np.empty(y.shape)
    x_values = np.empty(x.shape)
    n = len(x_values)
    for i in range(n):
        rand_index = random.randint(0, n - 1)
        x_values[i] = x[rand_index]
        y_values[i] = y[rand_index]
    return x_values, y_values


def get_best_variant(predicted):
    real_predicted = []
    for i in range(len(predicted[0])):
        all_results = {}
        for j in range(len(predicted)):
            if predicted[j][i] in all_results:
                all_results[predicted[j][i]] += 1
            else:
                all_results[predicted[j][i]] = 1
        ans, max_ = 0, 0
        for k in all_results:
            if all_results[k] > max_:
                max_ = all_results[k]
                ans = k
        real_predicted.append(ans)
    return real_predicted


def find_accuracy_for_unlimited_depth():
    for i in range(1, 22):
        num = str(i).rjust(2, '0')
        data_train = pd.read_csv("data/" + num + "_train.csv")
        data_test = pd.read_csv("data/" + num + "_test.csv")
        x, y = split_data(data_train)
        x_test, y_test = split_data(data_test)
        random_forest = []
        for _ in range(50):
            decisionTree = DecisionTreeClassifier(criterion="entropy", splitter="best" ,max_features='sqrt')
            x_, y_ = make_set_with_repetitions(x, y)
            decisionTree.fit(x_, y_)
            random_forest.append(decisionTree)
        predict_train = get_best_variant([decisionTree.predict(x) for decisionTree in random_forest])
        predict_test = get_best_variant([decisionTree.predict(x_test) for decisionTree in random_forest])
        test_acc = sum([1 if y_test[i] == predict_test[i] else 0 for i in range(len(y_test))]) / len(y_test)
        train_acc = sum([1 if y[i] == predict_train[i] else 0 for i in range(len(y))]) / len(y)
        print("Accuracy for " + num + " dataset:")
        print("   Train: " + str(train_acc))
        print("   Test: " + str(test_acc))
        print("<" + "=" * 32 + ">")


def main():
    find_accuracy_for_unlimited_depth()


if __name__ == '__main__':
    main()