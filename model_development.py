import emnist
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
import pandas as pd
from sklearn.model_selection import train_test_split
import visualizers


def extract_letters_from_dataset(dataset, labels, letter_numbers):
    mask = np.isin(labels, letter_numbers)
    dataset = dataset[mask]
    labels = labels[mask] - 1

    return dataset, labels


def load_letters_datasets(show_histogram=False):
    # Import more samples for letter i
    emnist_train_set, emnist_train_labels = emnist.extract_training_samples("letters")
    emnist_test_set, emnist_test_labels = emnist.extract_test_samples("letters")

    emnist_train_set, emnist_train_labels = extract_letters_from_dataset(emnist_train_set, emnist_train_labels, [9])
    emnist_test_set, emnist_test_labels = extract_letters_from_dataset(emnist_test_set, emnist_test_labels, [9])

    data = pd.read_csv("datasets/A_Z Handwritten Data.csv").astype('float32')

    letters = data.drop('0', axis=1)
    labels = data['0']

    train_x, test_x, train_y, test_y = train_test_split(letters, labels, test_size=0.2)
    train_x = np.reshape(train_x.values, (train_x.shape[0], 28, 28))
    test_x = np.reshape(test_x.values, (test_x.shape[0], 28, 28))

    # Concatenate the handwritten data with the i letter data
    train_x = np.concatenate((train_x, emnist_train_set), axis=0)
    test_x = np.concatenate((test_x, emnist_test_set), axis=0)
    train_y = np.concatenate((train_y, emnist_train_labels), axis=0)
    test_y = np.concatenate((test_y, emnist_test_labels), axis=0)

    train_x[train_x > 50] = 255
    train_x[train_x <= 50] = 0
    test_x[test_x > 50] = 255
    test_x[test_x <= 50] = 0

    if show_histogram:
        visualizers.show_dataset_histogram(np.concatenate((train_y, test_y), axis=0))

    return train_x, train_y, test_x, test_y


# Scale images from 0-255 to 0-1
def scale_images(image_dataset, max_pixel_value=255):
    image_dataset = image_dataset / max_pixel_value
    return image_dataset


# Creates the model
def create_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(27, activation="softmax"))
    return model


# Compiles the model
def compile_model(model):
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# Fit the model
def fit_model(model, training_set, training_labels, epochs):
    model.fit(training_set, training_labels, epochs=epochs, shuffle=True)
    return model


# Evaluate the model
def test_model(model, test_set, test_labels):
    return model.evaluate(test_set, test_labels, verbose=2)


def create_and_fit_model_with_params(train_set, train_labels):
    model = create_model()
    model = compile_model(model)
    model = fit_model(model, train_set, train_labels, epochs=10)

    return model


def create_train_save_letter_model(path_to_save="models/letter_recognition_model"):
    train_set, train_labels, test_set, test_labels = load_letters_datasets()

    train_set = scale_images(train_set)
    train_set = train_set.reshape(train_set.shape[0], train_set.shape[1], train_set.shape[2], 1)

    test_set = scale_images(test_set)
    test_set = test_set.reshape(test_set.shape[0], test_set.shape[1], test_set.shape[2], 1)

    model = create_and_fit_model_with_params(train_set, train_labels)
    print(test_model(model, test_set, test_labels))
    model.save(path_to_save)
