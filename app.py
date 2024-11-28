from PIL import Image, ExifTags
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import numpy as np
from vlad import VLAD
from bof import BOF
from fisher_vector import FisherVector
from adc import ADC
import io
import os
import base64
import cv2
app = Flask(__name__)
CORS(app)


def fix_image_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(orientation, None)
            if orientation == 3:  # Rotate 180 degrees
                image = image.rotate(180, expand=True)
            elif orientation == 6:  # Rotate 270 degrees (clockwise)
                image = image.rotate(270, expand=True)
            elif orientation == 8:  # Rotate 90 degrees (counterclockwise)
                image = image.rotate(90, expand=True)
    except AttributeError:
        pass
    return image


def extract_sift_features2(image):

    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    key_points, descriptors = sift.detectAndCompute(image, None)
    return key_points, descriptors


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


@app.before_request
def load():
    g.vlad = VLAD(n_vocabs=1, norming='RN', lcs=True)
    g.bof = BOF()
    g.fish = FisherVector()
    g.adc = ADC()
    # model_re = [vlad, bof, fish]


@app.route('/')
def home():
    return "Hello world!"


@app.route('/query', methods=['POST'])
def query():
    data = request.json
    img_repre = data.get('img_repre')
    dataset = data.get('dataset')
    query = data.get('query')
    image_data = data.get('image_data')
    result, re, path_img_repre, path_query = check_exists(
        img_repre, query, dataset)

    if not result:
        return jsonify({'error': 'Không tìm được model tương thích'}), 500
    adc = adc_get_param(g.adc, path_query)

    if re == 1:
        vlad = vlad_get_param(g.vlad, path_img_repre)
        if vlad.k / adc.m > 2 or vlad.k / adc.m < 1:
            return jsonify({'error': 'Không tìm được model tương thích'}), 500
        print(vlad.databases.shape, vlad.k)
    elif re == 2:
        bof = bof_get_param(g.bof, path_img_repre)
        if bof.k / adc.k == 1 or bof.k / adc.k > 20:
            return jsonify({'error': 'Không tìm được model tương thích'}), 500
        print(bof.databases.shape, bof.k)
    elif re == 3:
        fish = fish_get_param(g.fish, path_img_repre)
        if (fish.n_components / adc.m > 2) or (fish.n_components / adc.m < 1):
            return jsonify({'error': 'Không tìm được model tương thích'}), 500
        print(fish.databases.shape, fish.n_components)
    if dataset == 1:
        import json
        from ultis import get_all_image_paths, extract_sift_features
        image_directory = "datasets/INRIA/images/"

        image_paths = get_all_image_paths(image_directory)
        with open('datasets/INRIA/groundtruth.json', 'r') as file:
            data = json.load(file)
            for key in data:
                new_similar = []
                for img in data[key]['similar']:
                    for idx, path in enumerate(image_paths):
                        if img in path:
                            new_similar.append(idx)
                data[key]['similar'] = new_similar

        outputs = []
        for key in data:
            path = 'datasets/INRIA/images/' + data[key]['query']
            key_points, descriptors = extract_sift_features(path)
            if re == 1:
                query_vlad = vlad.transform([descriptors])

                probabilities = adc.predict_proba(query_vlad)
            elif re == 2:
                query_bof = bof.transform([descriptors])
                probabilities = adc.predict_proba(query_bof)
            elif re == 3:
                query_fish = fish.transform([descriptors])
                probabilities = adc.predict_proba(query_fish)
            sorted_indices = np.argsort(probabilities)

            output_per_query = []
            for i, index in enumerate(sorted_indices):
                print(
                    f"Original index: {index}, Probability: {probabilities[index]}")
                if i > 0:
                    output_per_query.append(index)

                    if len(output_per_query) == 30:
                        break

            outputs.append(output_per_query)

        from metrics.mAP import compute_map
        results = compute_map(data, outputs)
        print(results)
    else:
        from ultis import get_all_image_paths, extract_sift_features
        image_directory = "datasets/INRIA/images/"

        image_paths = get_all_image_paths(image_directory)
        from metrics.score4 import calculate_gr_score
        print('Running')
        similarity_matrix = []
        for path in image_paths:
            key_points, descriptors = extract_sift_features(path)

            if re == 1:
                query_vlad = vlad.transform([descriptors])
                probabilities = adc.predict_proba(query_vlad)
            elif re == 2:
                query_bof = bof.transform([descriptors])
                probabilities = adc.predict_proba(query_bof)
            elif re == 3:
                query_fish = fish.transform([descriptors])
                probabilities = adc.predict_proba(query_fish)

            similarity_matrix.append(probabilities)

        result = calculate_gr_score(np.array(similarity_matrix))
        print(result)
    try:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image = fix_image_orientation(image)
        _, descriptors = extract_sift_features2(image)

    except Exception as e:
        return jsonify({'error': 'Lỗi gì đó'}), 500

    return jsonify(data)


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)
