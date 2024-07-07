import numpy as np

from operators import TargetVectorDifferenceOperator, FirstOrderDifferenceOperator


class TargetPortfolioValueObjectiveFunction:
    """
    This objective function considers the operator TargetVectorDifferenceOperator applied to the total portfolio value.

    Denote the total portfolio value vector as V = [1, v1, v2, ..., vn], and the TargetVectorDifferenceOperator as D.

    The product D @ V computes the difference between the total portfolio value and the target portfolio value. Each
    component is the difference between the actual value and the target value for that day. We want to minimize these
    components, which can be done by minimizing the norm of the vector D @ V.

    The norm of the vector is denoted as ||D @ V|| = sqrt((D @ V)^T @ (D @ V)) = sqrt(V^T @ D^T @ D @ V).

    The gradient then is given by D^T @ D @ V / ||D @ V||, which we can use for minimization.

    This expression is more complicated than it appears since each component of V is actually the sum over all assets,
    so to compute the gradient, we need to do computations on all the underlying assets.
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
    This objective function considers the FirstOrderDifferenceOperator applied to the unit vectors independently.

    Denote the unit vectors as U = [u1, u2, ..., un], and the FirstOrderDifferenceOperator as K.

    The product K @ U computes the first order difference of each unit vector. We want to minimize these differences,
    which can be done by minimizing the norm of the vector K @ U.

    The norm of the vector is denoted as ||K @ U|| = sqrt((K @ U)^T @ (K @ U)) = sqrt(U^T @ K^T @ K @ U).

    The gradient then is given by K^T @ K @ U / ||K @ U||, which we can use for minimization.
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
