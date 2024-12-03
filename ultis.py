import os
import cv2
import numpy as np
from PIL import Image


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


def extract_hessian_sift_features(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sobel_xx = cv2.Sobel(image, cv2.CV_64F, 2, 0, ksize=3)
    sobel_yy = cv2.Sobel(image, cv2.CV_64F, 0, 2, ksize=3)
    sobel_xy = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=3)

    # Compute determinant and trace for the Hessian matrix
    det_hessian = sobel_xx * sobel_yy - sobel_xy**2
    trace_hessian = sobel_xx + sobel_yy
    print('Det', det_hessian.shape, np.mean(det_hessian),
          np.max(det_hessian), np.min(det_hessian))
    # Filter potential keypoints based on determinant and trace
    keypoints = np.where(det_hessian > 7000)  # threshold for robustness
    # Visualize keypoints
    keypoint_coords = [cv2.KeyPoint(
        float(pt[1]), float(pt[0]), 1) for pt in zip(*keypoints)]
    sift = cv2.SIFT_create()
    _, descriptors = sift.compute(image, keypoint_coords)
    return _, descriptors


def rotate(image_path, angle=0):
    if angle == 0 or angle == 360:
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = Image.open(image_path)
    rotated_image = image.rotate(angle, expand=True)
    cv_image = cv2.cvtColor(np.array(rotated_image), cv2.COLOR_BGR2RGB)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
    # cv2.imshow("Rotated Image", cv_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cv_image


def extract(image):
    sift = cv2.SIFT_create()
    key_points, descriptors = sift.detectAndCompute(image, None)
    return key_points, descriptors
