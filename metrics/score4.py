import numpy as np


def calculate_gr_score(similarity_matrix, group_size=4):
    n = similarity_matrix.shape[0]
    score = 0

    for i in range(n):
        scores = similarity_matrix[i, :]

        sorted_indices = np.argsort(scores)

        relevant_count = 0
        for rank in range(group_size):
            if sorted_indices[rank] // group_size == i // group_size:
                relevant_count += 1

        score += relevant_count

    return score / n
