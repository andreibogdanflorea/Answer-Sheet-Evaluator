import numpy as np
import string
import image_processing as imgproc
import tensorflow as tf
import model_development


def get_right_answers():
    return ['A', 'B', 'D', 'D', 'F', 'D', 'E', 'A', 'B', 'E',
            'B', 'B', 'C', 'E', 'C', 'D', 'A', 'A', 'D', 'F']


def create_class_to_letter_dict():
    class_to_letter = {}
    for idx, letter in enumerate(string.ascii_uppercase):
        class_to_letter[idx] = letter
    class_to_letter[26] = ' '

    return class_to_letter


def get_predictions(model, images_arr):
    results = []
    for im_arr in images_arr:
        if 0.99 * im_arr.size <= len(im_arr[im_arr == 0]):
            results.append(26)
            continue
        testing_arr = np.expand_dims(im_arr, axis=0)
        results.append(np.argmax(model.predict(testing_arr), axis=-1)[0])

    return predictions_to_letters(results)


def predictions_to_letters(predictions):
    class_to_letter = create_class_to_letter_dict()
    predicted_letters = [class_to_letter.get(predicted_class) for predicted_class in predictions]
    return predicted_letters


def get_predictions_ans(model, ans_arr):
    results = []
    for im_arr in ans_arr:
        if 0.99 * im_arr.size <= len(im_arr[im_arr == 0]):
            results.append(26)
            continue
        testing_arr = np.expand_dims(im_arr, axis=0)
        results.append(np.argmax(model.predict(testing_arr)[0][0:6], axis=-1))

    return predictions_to_letters(results)


def compute_score(participant_answers):
    score = 0
    correct_answers = get_right_answers()
    for i, answer in enumerate(participant_answers):
        if answer == correct_answers[i]:
            score += 5

    return score


def get_name_and_answers(path):
    model = tf.keras.models.load_model("models/best_trained_model")

    fname_squares, lname_squares, ans_squares = imgproc.extract_all_squares_from_image(
        path=path)
    fname_arr = imgproc.rgbimagelist_to_grey_nparray(fname_squares)
    lname_arr = imgproc.rgbimagelist_to_grey_nparray(lname_squares)
    ans_arr = imgproc.rgbimagelist_to_grey_nparray(ans_squares)

    fname_arr = np.array([imgproc.filter_image(image) for image in fname_arr])
    lname_arr = np.array([imgproc.filter_image(image) for image in lname_arr])
    ans_arr = np.array([imgproc.filter_image(image) for image in ans_arr])

    fname_arr = model_development.scale_images(fname_arr)
    fname_arr = fname_arr.reshape(fname_arr.shape[0], fname_arr.shape[1], fname_arr.shape[2], 1)

    lname_arr = model_development.scale_images(lname_arr)
    lname_arr = lname_arr.reshape(lname_arr.shape[0], lname_arr.shape[1], lname_arr.shape[2], 1)

    ans_arr = model_development.scale_images(ans_arr)
    ans_arr = ans_arr.reshape(ans_arr.shape[0], ans_arr.shape[1], ans_arr.shape[2], 1)

    fname_results = get_predictions(model, fname_arr)
    lname_results = get_predictions(model, lname_arr)
    ans_results = get_predictions_ans(model, ans_arr)

    first_name = "".join(fname_results).strip()
    last_name = "".join(lname_results).strip()

    return first_name, last_name, ans_results
