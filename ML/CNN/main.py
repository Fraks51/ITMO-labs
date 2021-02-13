import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import random

kernel_shape = [(1, 1), (3, 3), (5, 5)]
filters = [7, 14, 28, 56]
activation_func = ['relu', 'tanh']


def reshaper(X):
    X = X.reshape(list(X.shape) + [1])
    return X


def get_random_param():
    kernel = random.choice(kernel_shape)
    filters_number = random.choice(filters)
    activation = random.choice(activation_func)
    return kernel, filters_number, activation


def get_random_pooling():
    return layers.MaxPooling2D(random.choice([(2, 2), (3, 3)]))


def main():
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0


    best_accur = 0
    best_model = None

    train_images = reshaper(train_images)
    test_images = reshaper(test_images)
    test_labels = reshaper(test_labels)
    train_labels = reshaper(train_labels)

    for i in range(3):
        for _ in range(20):
            model = models.Sequential()
            kernel, filters_number, activation = get_random_param()
            model.add(layers.Conv2D(filters_number, kernel, padding="same", activation=activation, input_shape=(28, 28, 1)))
            model.add(get_random_pooling())
            print("Layer number {}: filters {}, kernel {}, activ {}".format(0, filters_number, kernel, activation))
            for _ in range(i):
                kernel, filters_number, activation = get_random_param()
                model.add(layers.Conv2D(filters_number, kernel, padding="same", activation=activation))
                model.add(get_random_pooling())
            model.add(layers.Flatten())
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(10, activation='softmax'))

            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy'])

            model.fit(train_images, train_labels, epochs=7)
            _, test_acc = model.evaluate(test_images, test_labels, verbose=0, use_multiprocessing=True)
            print("Accuracy: {}".format(test_acc))
            model.summary()
            if test_acc > best_accur:
                best_model = model
                best_accur = test_acc

    print("Best Accuracy: {}".format(best_accur))
    best_model.summary()


if __name__ == '__main__':
    main()

