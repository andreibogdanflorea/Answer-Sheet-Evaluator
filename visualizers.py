import matplotlib.pyplot as plt
import numpy as np


def show_letter(letter_image):
    plt.imshow(letter_image, cmap=plt.cm.Greys)
    plt.show()


def show_dataset_histogram(labels):
    train_yint = np.int0(labels)
    count = np.zeros(26, dtype='int')
    for i in train_yint:
        count[i] += 1

    word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
                 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
                 23: 'X', 24: 'Y', 25: 'Z'}

    alphabets = [i for i in word_dict.values()]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.barh(alphabets, count)

    plt.xlabel("Number of elements ")
    plt.ylabel("Alphabets")
    plt.grid()
    plt.show()
