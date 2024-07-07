import numpy as np


class DataGenerator:

    def __init__(self):
        self.rng = np.random.default_rng()

    def generate_random_normalized_matrix(self, num_of_rows, num_of_cols, axis=0):
        """
        Generates a random matrix such that the sum of the columns or rows is 1.
        :param num_of_rows:
        :param num_of_cols:
        :param axis: column-wise normalization if 0, row-wise normalization if 1
        :return:
        """
        matrix = self.rng.normal(10, 2, [num_of_rows, num_of_cols])
        return matrix / matrix.sum(axis=axis, keepdims=True)

    def generate_random_walk(self, initial_values, steps, distribution='normal', mu=0, sigma=1, lam=1):
        """
        Generate random walks from specified distributions.
        :param initial_values: List of initial values for the random walk.
        :param steps: Number of steps in the random walk.
        :param distribution: Type of distribution to use ('normal' or 'poisson').
        :param mu: Mean for the normal distribution (default is 0).
        :param sigma: Standard deviation for the normal distribution (default is 1).
        :param lam: Rate parameter for the Poisson distribution (default is 1).

        Returns:
        list: A list containing the random walks for each initial value.
        """
        random_walks = []

        for initial_value in initial_values:
            if distribution == 'normal':
                increments = self.rng.normal(mu, sigma, steps)
            elif distribution == 'poisson':
                increments = self.rng.poisson(lam, steps)
                signs = self.rng.choice([-1, 1], size=steps)
                increments = increments * signs
            else:
                raise ValueError("Unsupported distribution type. Use 'normal' or 'poisson'.")

            walk = np.cumsum(increments)
            walk = initial_value + walk
            random_walks.append(walk)

        return random_walks

    def generate_random_price_vectors(self, num_of_assets, num_of_days, start_price_range=(75, 100),
                                      distribution='poisson'):
        """
        Price vectors are of length num_of_days + 1. The first element of each price vector is set to 1/num_of_assets.
        The remaining elements are generated using random walks.
        :param num_of_assets:
        :param num_of_days:
        :param start_price_range: generates uniform random start prices in this range
        :param distribution: the type of distribution used for generating random walks
        :return:
        """
        price_vectors = np.full((num_of_assets, num_of_days + 1), 1 / num_of_assets)
        start_prices = self.rng.integers(*start_price_range, num_of_assets)
        random_walks = self.generate_random_walk(start_prices, num_of_days, distribution)
        for i, random_walk in enumerate(random_walks):
            price_vectors[i, 1:] = random_walk
        return price_vectors

    def generate_random_portfolio_value_vector(self, num_of_days, start_value=10000, distribution='normal',
                                               mu=0, sigma=100, lam=1):
        """
        Portfolio value vectors are of length num_of_days + 1. The first element of each portfolio value vector is set
        to one, and the remaining elements are generated using start value and a random walk.
        :param num_of_days:
        :param start_value:
        :param distribution:
        :param mu:
        :param sigma:
        :param lam:
        :return:
        """
        initial_values = [start_value]
        value_vector = np.ones(num_of_days + 1)
        random_walk = self.generate_random_walk(initial_values, num_of_days, distribution, mu, sigma, lam)[0]
        value_vector[1:] = random_walk
        return value_vector
