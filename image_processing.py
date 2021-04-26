from PIL import Image
import numpy as np
import cv2


def import_answersheet_image(path, show_image=False):
    im = Image.open(path)
    if show_image:
        im.show()
    return im


def filter_image(image):
    img_copy = image.copy()
    img_copy = cv2.GaussianBlur(img_copy, (3, 3), 0)
    _, img_thresh = cv2.threshold(img_copy, 50, 255, cv2.THRESH_BINARY)
    img_final = cv2.resize(img_thresh, (28, 28))
    return img_final


def paste_images_top_bottom(im1, im2):
    dest = Image.new("RGB", (im1.width, im1.height + im2.height))
    dest.paste(im1, (0, 0))
    dest.paste(im2, (0, im1.height))
    return dest


def extract_needed_parts_old(image, new_size=(600, 850)):
    new_image = image.resize(new_size)

    firstname_rect = (60, 165, 543, 196)
    lastname_rect = (60, 263, 543, 294)

    firstname_squares = new_image.crop(firstname_rect)
    lastname_squares = new_image.crop(lastname_rect)

    first_column_answers_rectangle = (161, 378, 217, 744)
    second_column_answers_rectangle = (484, 378, 540, 744)

    first_column_ans = new_image.crop(first_column_answers_rectangle)
    second_column_ans = new_image.crop(second_column_answers_rectangle)
    column_ans = paste_images_top_bottom(first_column_ans, second_column_ans)

    return firstname_squares, lastname_squares, column_ans


def extract_all_squares_from_image_old(path="images/answerSheet.png"):
    image = import_answersheet_image(path)
    firstname_squares, lastname_squares, ans_column = extract_needed_parts(image)

    def split_name_squares_to_list_old(name_squares):
        count = 12
        (top_left_x, top_left_y) = (2, 2)
        (offset_x, offset_y) = (35, 27)
        next_x = 40

        squares_list = []
        for i in range(count - 2):
            square = name_squares.crop((top_left_x + i * next_x, top_left_y,
                                        top_left_x + offset_x + i * next_x, top_left_y + offset_y))
            squares_list.append(square)

        for i in range(count - 2, count):
            square = name_squares.crop((top_left_x + i * next_x + 3, top_left_y,
                                        top_left_x + offset_x + i * next_x + 3, top_left_y + offset_y))
            squares_list.append(square)

        return list(map(lambda im: im.resize((32, 32)).crop((2, 2, 30, 30)), squares_list))

    def split_ans_column_to_list_old(ans_col):
        count = 20
        (top_left_x, top_left_y) = (2, 3)
        (offset_x, offset_y) = (52, 31)
        next_y = [0, 36, 37, 36, 37, 37, 36, 36, 37, 36, 38, 36, 37, 36, 37, 37, 36, 36, 37, 37]
        for i in range(1, len(next_y)):
            next_y[i] = next_y[i] + next_y[i - 1]

        squares_list = []
        for i in range(count):
            square = ans_col.crop((top_left_x, top_left_y + next_y[i],
                                   top_left_x + offset_x, top_left_y + offset_y + next_y[i]))
            squares_list.append(square)

        return list(map(lambda im: im.resize((32, 32)).crop((2, 2, 30, 30)), squares_list))

    firstname_square_list = split_name_squares_to_list_old(firstname_squares)
    lastname_square_list = split_name_squares_to_list_old(lastname_squares)
    ans_column_square_list = split_ans_column_to_list_old(ans_column)

    return firstname_square_list, lastname_square_list, ans_column_square_list


def extract_needed_parts(image, new_size=(1653, 2377)):
    new_image = image.resize(new_size)

    firstname_rect = (190, 482, 1480, 570)
    lastname_rect = (190, 749, 1480, 836)

    firstname_squares = new_image.crop(firstname_rect)
    lastname_squares = new_image.crop(lastname_rect)

    first_column_answers_rectangle = (460, 1071, 604, 2062)
    second_column_answers_rectangle = (1322, 1071, 1470, 2062)

    first_column_ans = new_image.crop(first_column_answers_rectangle)
    second_column_ans = new_image.crop(second_column_answers_rectangle)
    column_ans = paste_images_top_bottom(first_column_ans, second_column_ans)

    firstname_squares.save("images/fnamesquares.png")
    lastname_squares.save("images/lnamesquares.png")
    column_ans.save("images/column_ans.png")

    return firstname_squares, lastname_squares, column_ans


def extract_all_squares_from_image(path="images/answerSheet.png"):
    image = import_answersheet_image(path)
    firstname_squares, lastname_squares, ans_column = extract_needed_parts(image)

    def split_name_squares_to_list(name_squares):
        top_y = 3
        (size_x, size_y) = (80, 80)
        x_list = [8, 118, 220, 336, 443, 552, 655, 766, 874, 981, 1091, 1201]

        squares_list = [name_squares.crop((x, top_y, x + size_x, top_y + size_y)) for x in x_list]

        return list(map(lambda im: im.resize((32, 32)).crop((2, 2, 30, 30)), squares_list))

    def split_ans_column_to_list(ans_col):
        top_x = 30
        (size_x, size_y) = (85, 85)
        y_list = [5, 103, 204, 302, 401, 501, 600, 698, 798, 896, 994, 1096, 1194, 1290,
                  1389, 1488, 1587, 1685, 1786, 1887]

        squares_list = [ans_col.crop((top_x, y, top_x + size_x, y + size_y)) for y in y_list]

        return list(map(lambda im: im.resize((32, 32)).crop((2, 2, 30, 30)), squares_list))

    firstname_square_list = split_name_squares_to_list(firstname_squares)
    lastname_square_list = split_name_squares_to_list(lastname_squares)
    ans_column_square_list = split_ans_column_to_list(ans_column)

    return firstname_square_list, lastname_square_list, ans_column_square_list


def rgbimagelist_to_grey_nparray(images_list):
    images_arr = np.array([])
    for image in images_list:
        image_arr = 255 - np.asarray(image.convert("L"))
        images_arr = np.append(images_arr, image_arr)

    return images_arr.reshape((len(images_list), 28, 28))
