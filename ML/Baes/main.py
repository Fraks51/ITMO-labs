import os
import math

class BayesClassifier:

    def __init__(self, alpha, lambdas):
        self.alpha = alpha
        self.lambdas = lambdas
        self.k = 2
        self.counter = [0 for _ in range(self.k)]
        self.class_and_word = dict()

    def fit(self, dataset):
        for i in range(len(dataset)):
            words, clazz = dataset[i]
            self.counter[clazz] += 1
            for word in set(words):
                if word in self.class_and_word:
                    self.class_and_word[word][clazz] += 1
                else:
                    self.class_and_word[word] = [0 for _ in range(self.k)]
                    self.class_and_word[word][clazz] += 1
        self.ps = \
            [{word: (self.class_and_word[word][clazz] + self.alpha) / (self.counter[clazz] + self.alpha * 2)
              for word in self.class_and_word}
             for clazz in range(self.k)]
        self.precount = [sum([math.log(1 - self.ps[clazz][word]) for word in self.class_and_word])
                         for clazz in range(self.k)]

    def predict(self, test_data, get_params=False):
        result = []
        result_classes = []
        n = sum(self.counter)
        for words, _ in test_data:
            classes = []
            for clazz in range(self.k):
                pr = self.precount[clazz]
                for word in set(words):
                    if word in self.class_and_word:
                        pr += math.log(self.ps[clazz][word]) - math.log(1 - self.ps[clazz][word])
                classes.append(self.lambdas[clazz] + math.log(self.counter[clazz] / n) + pr)
            result.append(classes.index(max(classes)))
            result_classes.append(classes)
        if get_params:
            return result_classes
        else:
            return result

    def binary_predict(self, test_data):
        result = []
        n = sum(self.counter)
        for words, _ in test_data:
            classes = []
            for clazz in range(self.k):
                pr = self.precount[clazz]
                for word in set(words):
                    if word in self.class_and_word:
                        pr += math.log(self.ps[clazz][word]) - math.log(1 - self.ps[clazz][word])
                classes.append(self.lambdas[clazz] + math.log(self.counter[clazz] / n) + pr)
            result.append(float(classes[0] / sum(classes)))
        return result


def n_gram_from_lists(words: list, n: int):
    n_grams = []
    for i in range(0, len(words) + 1 - n):
        n_grams.append("_".join(words[i:i + n]))
    return n_grams


def made_n_gram(path: str, name: str, n: int):
    clazz = 1 if "spmsg" in name else 0  # 1 if msg is a spam, else 0
    file_reader = open(path + "/" + name)
    line = file_reader.readline().split()
    subject_n_grams = n_gram_from_lists(line[1:], n)
    _ = file_reader.readline()
    body_n_grams = n_gram_from_lists(file_reader.readline().split(), n)
    n_grams = subject_n_grams + body_n_grams
    return n_grams, clazz


def get_grams_sets(n: int):
    grams_sets = []
    for i in range(1, 11):
        path = "data/part" + str(i)
        part_set_of_grams = [made_n_gram(path, name, n)
                             for name in os.listdir(path=path)]
        grams_sets.append(part_set_of_grams)
    return grams_sets


def get_accuracy(grams_sets, alphas, lambdas):
    accuracy = 0
    n = 0
    for i in range(10):
        train_data = []
        for j in range(10):
            if j != i:
                train_data = train_data + grams_sets[j]
        test_data = grams_sets[i]
        n += len(test_data)
        model = BayesClassifier(alphas, lambdas)
        model.fit(train_data)
        predicted = model.predict(test_data)
        accuracy += sum([1 if predicted[i] == test_data[i][1] else 0
                         for i in range(len(test_data))])
    return accuracy / n


def find_params(n: int, lambdas):
    grams_sets = get_grams_sets(n)
    for alphas in [10 ** (-6), 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]:
        accuracy = get_accuracy(grams_sets, alphas, lambdas)
        print("Alphas = " + str(alphas))
        print("Accuracy : " + str(accuracy))


def find_lambda():
    grams_sets = get_grams_sets(2)
    good_lambda = 1
    for i in range(10):
        train_data = []
        for j in range(10):
            if j != i:
                train_data = train_data + grams_sets[j]
        test_data = grams_sets[i]
        model = BayesClassifier(0.0001, [1, 1])
        model.fit(train_data)
        predicted = model.predict(test_data, get_params=True)
        for j in range(len(predicted)):
            _, y = test_data[j]
            if y == 0 and predicted[i][0] < predicted[i][1]:  # l[0] + (a - 1) > b => l[0] > b - a + 1
                good_lambda = max((predicted[i][1] + 1) - predicted[i][0] + 10 ** (-10), good_lambda)
    return [good_lambda, 1]


'[1916.29, 1]'


def main():
    find_params(2, [1, 1])


def get_x_y_for_plot():
    grams_sets = []
    for i in range(1, 11):
        path = "data/part" + str(i)
        part_set_of_grams = [made_n_gram(path, name, 2)
                             for name in os.listdir(path=path)]
        grams_sets.append(part_set_of_grams)
    x = []
    y = []
    for i in range(1, 1916, 20):
        x.append(i)
        print(i)
        y.append(get_accuracy(grams_sets, 0.0001, [i, 1]))
    print(x)
    print(y)


if __name__ == '__main__':
    print(find_lambda())
    # get_x_y_for_plot()
