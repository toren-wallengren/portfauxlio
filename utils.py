import numpy as np


def generate_normalized_matrix(N, M, axis=0):
    rng = np.random.default_rng()
    matrix = np.maximum(rng.normal(10, 2, [N, M]), 0)
    return matrix / matrix.sum(axis=axis, keepdims=True)


def generate_random_unit_vectors(price_vectors, total_value_vector):
    N, T = price_vectors.shape
    normalized_matrix = generate_normalized_matrix(N, T)
    result = total_value_vector * normalized_matrix / price_vectors
    result[:, 0] = 1
    return result
