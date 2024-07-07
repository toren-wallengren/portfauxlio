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
        :param target_vector:
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


class FirstOrderDifferenceOperator:
    """
    Operator that computes the first order difference of a vector. The matrix representation takes the form
    [[1, 0, 0, ..., 0],
    [-v0, 1, 0, ..., 0],
    [0, -1, 1, ..., 0],
    ...
    [0, 0, 0, ..., 1]]

    There is an optional parameter initial_value that can be used to force the gradient to pull the first element
    towards that fixed value.
    """

    def __init__(self, num_of_days, initial_value=1):
        self.matrix = np.eye(num_of_days) - np.eye(num_of_days, k=-1)
        self.matrix[1, 0] = -initial_value
        self.gram_matrix = self.matrix.T @ self.matrix

    def apply(self, vector):
        return self.matrix @ vector

    def apply_gram(self, vector):
        return self.gram_matrix @ vector
