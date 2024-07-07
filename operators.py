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

    def compute_gradient(self, vector):
        return 2 * self.apply_gram(vector)


class TargetPortfolioValuesOperator:
    """
    This operator is the TargetVectorDifferenceOperator applied to the total portfolio value.

    Since the portfolio value is derived from the underlying price and unit vectors, the gradient is computed by
    applying a whole family of linear operators to combinations of the unit vectors and summing them up.

    For each pair of assets i and j, the gradient contribution is computed as:
    P(i)*D.T*D*P(j)*u(j)

    where P(i) & P(j) are the price matrices of the assets, D is the target vector difference operator (and D.T is
    its transpose), and u(j) is the unit vector of asset j.

    The total gradient with respect to i is the sum of these contributions for all j (plus a normalization factor).
    """

    def __init__(self, target_portfolio_values, price_vectors):
        num_of_assets, num_of_days = price_vectors.shape
        self.tvd = TargetVectorDifferenceOperator(target_portfolio_values)
        self.price_vectors = [np.diag(price_vectors[i, :]) for i in range(num_of_assets)]
        self.operators = self._initialize_operators(num_of_assets)

    def _initialize_operators(self, num_of_assets):
        """
        We initialize all underlying matrices used in the gradient computation here, rather than redoing matrix
        calculations at each iteration. It is not clear whether this is the best strategy. We should investigate
        performance implications at some time in the future.
        :param num_of_assets:
        :return:
        """
        return [self.tvd.apply_gram(self.price_vectors[i]) for i in range(num_of_assets)]

    def compute_gradient(self, unit_vectors):
        gradient = np.zeros_like(unit_vectors)
        products = []
        for i, vector_1 in enumerate(unit_vectors):
            for j, vector_2 in enumerate(unit_vectors):
                if len(products) <= j:
                    # this product is needed for every i, so caching it here might improve performance
                    products.append(self.operators[j] @ vector_2)
                gradient[i, :] += self.price_vectors[i] @ products[j]

        value_vectors = [price_vector @ unit_vector for price_vector, unit_vector in
                         zip(self.price_vectors, unit_vectors)]
        total_portfolio_value = np.sum(value_vectors, axis=0)
        return gradient / np.linalg.norm(self.tvd.apply(total_portfolio_value))
