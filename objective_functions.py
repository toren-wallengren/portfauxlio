import numpy as np

from operators import TargetVectorDifferenceOperator, FirstOrderDifferenceOperator


class TargetPortfolioValueObjectiveFunction:
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


class FirstOrderUnitSmoothingObjectiveFunction:
    """
    This operator is the FirstOrderDifferenceOperator applied to the unit vectors.
    """

    def __init__(self, initial_unit_vectors):
        num_of_assets, num_of_days = initial_unit_vectors.shape
        self.fod = [FirstOrderDifferenceOperator(num_of_days, initial_value=initial_unit_vectors[i, 1])
                    for i in range(num_of_assets)]
        self.num_of_assets = num_of_assets

    def compute_gradient(self, unit_vectors):
        gradient = np.zeros_like(unit_vectors)
        for i in range(self.num_of_assets):
            unit_vector = unit_vectors[i, :]
            apply_gram = self.fod[i].apply_gram(unit_vector)
            apply_op = self.fod[i].apply(unit_vector)
            norm = np.linalg.norm(apply_op)
            grad = apply_gram / norm
            gradient[i, :] += grad
        return gradient