import numpy as np


class TargetVectorDifferenceOperator:
    """
    Operator that computes the difference between a vector and a target vector. The matrix representation takes the form
    [[1, 0, 0, ..., 0],
    [-v1, 1, 0, ..., 0],
    [-v2, 0, 1, ..., 0],
    ...
    [-vn, 0, 0, ..., 1]]
    where v1, v2, ..., vn are the components of the target vector.
    """

    def __init__(self, target_vector):
        """
        The actual target vector should be embedded in the last N-1 components of the target_value_vector. ie we assume
        it has the form [1, v1, v2, ..., vn].
        :param target_value_vector:
        """
        n = len(target_vector)
        self.matrix = np.zeros([n, n])
        for i in range(n):
            if i > 0:
                self.matrix[i, 0] = -target_vector[i]
            self.matrix[i, i] = 1
        self.gram_matrix = self.matrix.T @ self.matrix

    def apply(self, vector):
        return self.matrix @ vector

    def apply_gram(self, vector):
        return self.gram_matrix @ vector

    def compute_gradient(self, vector):
        return 2 * self.apply_gram(vector)


class TargetPortfolioValuesOperator:

    def __init__(self, target_portfolio_values, price_vectors):
        num_of_assets, num_of_days = price_vectors.shape
        self.tvd = TargetVectorDifferenceOperator(target_portfolio_values)
        self.price_vectors = [np.diag(price_vectors[i, :]) for i in range(num_of_assets)]
        self.operators = [[self.price_vectors[i] @ self.tvd.apply_gram(self.price_vectors[j])
                           for j in range(num_of_assets)] for i in range(num_of_assets)]

    def compute_gradient(self, unit_vectors):
        gradient = np.zeros_like(unit_vectors)
        for i, vector_1 in enumerate(unit_vectors):
            for j, vector_2 in enumerate(unit_vectors):
                gradient[i, :] += self.operators[i][j] @ vector_2

        value_vectors = [price_vector @ unit_vector for price_vector, unit_vector in
                         zip(self.price_vectors, unit_vectors)]
        total_portfolio_value = np.sum(value_vectors, axis=0)
        return gradient / np.linalg.norm(self.tvd.apply(total_portfolio_value))
