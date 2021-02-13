import numpy as np
import tensorflow as tf


class Node:
    def __init__(self, value):
        self.value = value
        self.childes = {}
        self.nodes = {}

    def inc(self, a):
        if a in self.childes:
            self.childes[a] += 1
        else:
            self.childes[a] = 0
            self.nodes[a] = Node(a)

    def get_greater(self):
        a = ''
        max_c = 0
        for i in self.childes:
            if self.childes[i] > max_c:
                max_c = self.childes[i]
                a = i
        return a


def reshaper(X):
    X = X.reshape(list(X.shape) + [1])
    return X


def build_mark_chain(X, Y):
    root = Node("")
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        this_node = root
        for c in x:
            this_node.inc(c)
            this_node = this_node.nodes[c]
        this_node.inc(y)
    return root


def predict_next_char(chain, y):
    this_node = chain
    for c in y:
        this_node = this_node.nodes[c]
    return this_node.get_greater()


def preprocessing_data(name):
    f = open("data/" + name, "r+")
    content = f.read()
    content = ''.join(filter(lambda c: not (c in [',', '(', ')', ':', '-', "'"]), content))
    content = content.lower()
    content = content.replace(';', '.').replace('\n', '.').replace('!', '.').replace('?', '.')
    return content


def one_hot(ys, n):
    Y = []
    for y in ys:
        zeros = [0 for _ in range(n)]
        zeros[y] = 1
        Y.append(np.array(zeros))
    return np.array(Y)


def main():
    train = preprocessing_data("bible.txt")
    all_chars = set(train)
    char2num = {c: i for i, c in enumerate(all_chars)}
    num2char = np.array(all_chars)
    sentences = train.split(".")
    seq_length = 50
    raw_X = []
    raw_y = []
    for i in range(len(sentences)):
        j = i + 1
        while len(sentences[i]) < 32 and j < len(sentences):
            sentences[i] += " " + sentences[j]
            j += 1
    K = 0
    for sen in sentences[:5000]:
        print(K)
        K += 1
        for i in range(0, len(sen) - seq_length):
            raw_X.append(np.array([char2num[c] / len(all_chars) for c in sen[i: i + seq_length]]))
            raw_y.append(char2num[sen[i + seq_length]])
    mark_chain = build_mark_chain(raw_X, raw_y)
    prefix = input("Type prefix:")
    for i in range(20):
        c = predict_next_char(mark_chain, prefix)
        print(c, end='')
        prefix.append(c)
        prefix = prefix[1:]
    #
    # raw_X = np.array(raw_X)
    # X = reshaper(raw_X)
    # Y = one_hot(raw_y, len(all_chars))
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Embedding(len(all_chars), 256, batch_input_shape=(X.shape[1], X.shape[2])) ,
    #     tf.keras.layers.LSTM(256),
    #     tf.keras.layers.Dense(Y.shape[1], activation='softmax')
    # ])
    # model.compile(loss='categorical_crossentropy', optimizer='adam')
    # model.fit(X, Y, epochs=20)
    # prefix = input("Type prefix:")
    # prefix = [char2num[c] / all_chars for c in prefix]
    # for i in range(200):
    #     x = reshaper(prefix)
    #     predict = model.predict(x)
    #     i = np.argmax(predict)
    #     print(num2char[i], end='')
    #     prefix.append(i)
    #     prefix = prefix[1:]


if __name__ == '__main__':
    main()
