import os
import cv2
import numpy as np


def get_all_image_paths(directory):
    """
    Retrieves all image paths from subdirectories of the given directory.

    Args:
        directory (str): Path to the dataset directory.

    Returns:
        list: List of image paths.
    """
    image_paths = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                image_paths.append(os.path.join(root, file))

    return image_paths


def extract_sift_features(image_path):
    """
    Extracts local SIFT features from an image.

    Args:
        image_path (str): Path to the image.

    Returns:
        tuple: Tuple containing key points and descriptors.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    key_points, descriptors = sift.detectAndCompute(image, None)
    return key_points, descriptors


def save_all_features(all_features, dataset):
    if not os.path.exists("checkpoint"):
        os.makedirs("checkpoint")
    np.savez_compressed("checkpoint/dataset_{}.npz".format(dataset),
                        *all_features)


def load_dataset(path):
    data = np.load(path)
    return [data[f"arr_{i}"] for i in range(len(data.files))]
