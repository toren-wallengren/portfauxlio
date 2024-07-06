import numpy as np


def generate_normalized_matrix(N, M, axis=0):
    matrix = np.random.rand(N, M)
    return matrix / matrix.sum(axis=axis, keepdims=True)


def generate_random_unit_vectors(price_vectors, total_value_vector):
    N, T = price_vectors.shape
    normalized_matrix = generate_normalized_matrix(N, T)
    result = total_value_vector * normalized_matrix / price_vectors
    result[:, 0] = 1
    return result


def generate_random_price_vectors(num_of_assets, num_of_days, start_price_range=(10, 50), price_step_range=(-2, 2)):
    start_prices = np.random.randint(*start_price_range, num_of_assets)
    price_steps = np.random.randint(*price_step_range, size=(num_of_assets, num_of_days - 1))
    price_vectors = np.ones((num_of_assets, num_of_days+1))/num_of_assets
    price_vectors[:, 1] = start_prices
    for day in range(1, num_of_days):
        price_vectors[:, day+1] = price_vectors[:, day] + price_steps[:, day - 1]
    return np.maximum(price_vectors, 0)


def build_total_value_operator(num_of_days, desired_total_value):
    n = num_of_days + 1
    D = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            D[i, 0] = -desired_total_value
        D[i, i] = 1
    return D


def build_local_value_operator(num_of_days, initial_value):
    n = num_of_days + 1
    K = np.zeros((n, n))
    for i in range(n):
        if i == 1:
            K[i, 0] = -initial_value
        if i > 1:
            K[i, i - 1] = -1
        K[i, i] = 1
    return K
