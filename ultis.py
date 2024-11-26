import os
import cv2


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
