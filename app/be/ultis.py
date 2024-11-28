import os
from ultis import get_all_image_paths
import base64


def check_exists(img_repre, query, dataset):
    map = ['vlad', 'bof', 'fish']

    re = [i+1 for i, value in enumerate(map) if value in img_repre][0]
    path_img_repre = 'checkpoint/' + img_repre + '_ds{}'.format(dataset)
    print(path_img_repre)

    if not os.path.exists(path_img_repre):
        return False, None, None, None

    path_query = 'checkpoint/' + query + \
        '_ds{}'.format(dataset) + '_re{}'.format(re)

    print(path_query)

    if not os.path.exists(path_query):
        return False, None, None, None

    path_dataset = 'checkpoint/' + 'dataset_{}.npz'.format(dataset)

    print(path_dataset)

    if not os.path.exists(path_dataset):
        return False, None, None, None
    return True, re, path_img_repre, path_query


def vlad_get_param(model, path):
    parts = path.split('_')
    k = int(parts[1][1:])
    print(k)
    model.k = k
    model.load_model(path + '/')
    return model


def bof_get_param(model, path):
    parts = path.split('_')
    k = int(parts[1][1:])
    model.k = k
    model.load_model(path + '/')
    return model


def fish_get_param(model, path):
    parts = path.split('_')
    n_components = int(parts[1][1:])
    model.n_components = n_components
    model.load_model(path + '/')
    return model


def adc_get_param(model, path):
    parts = path.split('_')
    k = int(parts[1][1:])
    m = int(parts[2][1:])
    model.k = k
    model.m = m
    model.load_model(path + '/')
    return model


def encode(path):
    with open(path, 'rb') as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
    return encoded_string


def extract_img_from_folder(output, dataset):
    if int(dataset) == 1:
        image_directory = "datasets/INRIA/images/"

    elif int(dataset) == 2:
        image_directory = "datasets/UKB/full"

    image_paths = get_all_image_paths(image_directory)
    out = []
    for id in output:
        for i, path in enumerate(image_paths):
            if i == id:
                out.append(encode(path))
    return out


def get_output(sorted_indices, dataset, topk=4):
    output_per_query = []
    for i, index in enumerate(sorted_indices):
        if i > 0:
            output_per_query.append(index)
            if len(output_per_query) == topk:
                break

    return extract_img_from_folder(output_per_query, dataset)
