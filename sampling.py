import cv2
import numpy as np
import json
import os

DATA_PATH = './data'

SAMPLES_PATH = './samples'


# Получение списка путей файлов с нужным расширением
def get_list_path(dir_path, file_filter):
    list_json_path = []

    for path_dir, name_dir, file_name in os.walk(dir_path):
        list_json_path = [file for file in file_name if file[-(len(file_filter)):] == file_filter]

    return list_json_path


# Просмотр изображения
def view_image(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0)


# Получение массива numpy с содержание контуров
def get_contours(json_file):
    contours = []
    shapes = json_file['shapes']

    for i in range(0, len(shapes)):
        contours.append(np.array((shapes[i]['points'])))

    return contours


# Обрезание и изменение размера изображения
def crop_resize_image(img, contours, img_h, img_w):
    x, y = [], []

    for contour_line in contours:
        x.append(contour_line[0])
        y.append(contour_line[1])

    x1, x2, y1, y2 = min(x), max(x), min(y), max(y)

    if x2 - x1 < y2 - y1:
        long_side = y2 - y1
    else:
        long_side = x2 - x1

    cropped_img = img[y1:y1+long_side, x1:x1+long_side]

    if y1 + long_side > img_h:
        a = np.zeros((y1 + long_side - img_h, cropped_img.shape[1], 3), dtype='uint8')
        a[:] = cropped_img[-1][-1]
        cropped_img = np.concatenate((cropped_img, a))

    if x1 + long_side > img_w:
        a = np.zeros((cropped_img.shape[0], x1 + long_side - img_w, 3), dtype='uint8')
        a[:] = cropped_img[-1][-1]
        cropped_img = np.concatenate((cropped_img, a), axis=1)

    cropped_img = cv2.resize(cropped_img, (128, 128))

    return cropped_img


def main():
    json_list = get_list_path(DATA_PATH, '.json')
    labels = {}
    num = 1
    index = 0

    for json_name in json_list:
        with open(os.path.join(DATA_PATH, json_name)) as f:
            json_file = json.load(f)
            shapes = json_file['shapes']
            img_h, img_w = json_file['imageHeight'], json_file['imageWidth']
            for i in range(0, len(shapes)):
                labels[index] = (shapes[i]['label'])
                index += 1

        contours = get_contours(json_file)

        img = cv2.imread(os.path.join(DATA_PATH, json_name.replace('json', 'png')))

        for i in range(0, len(contours)):
            cropped_image = crop_resize_image(img, contours[i], img_h, img_w)
            cv2.imwrite(f'{SAMPLES_PATH}/{num}.png', cropped_image)
            num += 1

    with open('labels.json', 'w') as f:
        f.write(json.dumps(labels))


if __name__ == '__main__':
    main()
