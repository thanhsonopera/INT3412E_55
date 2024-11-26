
from ultis import *
import argparse
import warnings
import json
from bof import BOF
from fisher_vector import FisherVector
from vlad import VLAD
from adc import ADC
from metrics.mAP import compute_map
from metrics.score4 import calculate_gr_score
import numpy as np
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(
        description="Aggregating local descriptors into a compact image representation")
    parser.add_argument("--data", "-d", type=int,
                        default=0, help="Type of data to use [INRIA, UKB]")
    parser.add_argument("--feature-extract", "-fe", type=int, default=0,
                        help="Feature extraction method to use [SIFT]")
    parser.add_argument("--vector-represent", "-vr", type=int, default=0,
                        help="Vector representation method to use [VLAD, BOF, FISHER]")
    parser.add_argument("--vector-to-query", "-vtq", type=int, default=0,
                        help="Vector to query method to use [ADC])")
    parser.add_argument("--cluster-represent", "-cr", type=int,
                        default=16, help="Number of clusters represent to use [16, 64, 1024, 20480]")
    parser.add_argument("--cluster-query", "-cq", type=int, default=256,
                        help="Number of clusters to query [256, 1024]")
    parser.add_argument("--num-subvectors", "-m", type=int, default=16,
                        help="Number of subvectors to use")

    args = parser.parse_args()
    return args


def train(args):
    if args.data == 0:
        image_directory = "datasets/INRIA/images/"
        num_samples = 1491
    elif args.data == 1:
        image_directory = "datasets/UKB/full/"
        num_samples = 10200
    else:
        raise ValueError("Error: Invalid dataset type.")
    image_paths = get_all_image_paths(image_directory)
    assert len(
        image_paths) == num_samples, f"Error: Expected {num_samples} samples, but got {len(image_paths)}."

    all_features = []
    if args.feature_extract == 0:
        dimension = 128
        for _, path in enumerate(image_paths):
            _, descriptors = extract_sift_features(path)

            if descriptors is not None:
                all_features.append(descriptors)
    else:
        raise ValueError("Error: Invalid feature extraction method.")

    print("All features extracted.")

    if args.vector_represent == 0:
        if args.cluster_represent in [16, 64]:
            represent = VLAD(
                k=args.cluster_represent, n_vocabs=1, norming="RN", lcs=True).fit(all_features)
        else:
            raise ValueError(
                "Error: Invalid number of clusters represent VLAD.")
    elif args.vector_represent == 1:
        if args.cluster_represent in [1024, 20480] and args.cluster_query == 0:
            represent = BOF(k=args.cluster_represent).fit(all_features)
        elif args.cluster_represent in [1000, 20000] and args.cluster_query == 1:
            represent = BOF(k=args.cluster_represent).fit(all_features)
        else:
            raise ValueError(
                "Error: Invalid number of clusters represent BOF.")
    elif args.vector_represent == 2:
        if args.cluster_represent in [16, 64]:
            represent = FisherVector(n_components=args.cluster_represent).fit(
                all_features)
        else:
            raise ValueError(
                "Error: Invalid number of clusters represent Fisher Vector.")
    else:
        raise ValueError("Error: Invalid vector representation method.")

    print("All features represented.")

    if args.vector_to_query == 0:
        if args.cluster_query in [256, 1024] and dimension % args.num_subvectors == 0:
            adc = ADC(m=args.num_subvectors, k=args.cluster_query)
            adc.fit(represent.databases)
        else:
            raise ValueError("Error: Invalid number of clusters query VLAD.")
    elif args.vector_to_query == 1:
        pass
    else:
        raise ValueError("Error: Invalid vector to query method.")

    print("All features to coded.")

    if args.data == 0:
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
            path = image_directory + data[key]['query']
            _, descriptors = extract_sift_features(path)

            query_encode = represent.transform([descriptors])
            probabilities = adc.predict_proba(query_encode)

            sorted_indices = np.argsort(probabilities)

            output_per_query = []
            for i, index in enumerate(sorted_indices):
                # print(
                #     f"Original index: {index}, Probability: {probabilities[index]}")
                if i > 0:
                    output_per_query.append(index)
                    # if len(output_per_query) == len(data[key]['similar']):
                    #     break
                    if len(output_per_query) == 30:
                        break

            outputs.append(output_per_query)

        score = compute_map(data, outputs)
        print(f"mAP: {score} | vectors representing {args.vector_represent} /
              | vectors to query {args.vector_to_query} | clusters representing {args.cluster_represent} /
              | clusters to query {args.cluster_query} | number subvector {args.num_subvectors}")

    elif args.data == 1:
        similarity_matrix = []
        for path in image_paths:
            _, descriptors = extract_sift_features(path)

            query_encode = represent.transform([descriptors])
            probabilities = adc.predict_proba(query_encode)

            similarity_matrix.append(probabilities)

        score = calculate_gr_score(np.array(similarity_matrix))
        print(f"Score/4: {score} | vectors representing {args.vector_represent} /
              | vectors to query {args.vector_to_query} | clusters representing {args.cluster_represent} /
              | clusters to query {args.cluster_query} | number subvector {args.num_subvectors}")


if __name__ == "__main__":
    args = get_args()
    train(args)
