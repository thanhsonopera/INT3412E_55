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
from ultis import *
from app.be.ultis import *
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

    try:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image = fix_image_orientation(image)
        _, descriptors = extract_sift_features2(image)
        print('Extracting features')
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

        if int(dataset) == 1:
            output_encode_per_query = get_output(
                sorted_indices, dataset, topk=10)
            # print(output_encode_per_query)
        else:
            output_encode_per_query = get_output(sorted_indices, dataset)
            # print(output_encode_per_query)

        return jsonify({'images': output_encode_per_query})
    except Exception as e:
        return jsonify({'error': 'Lỗi gì đó'}), 500

    return jsonify(data)


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)
