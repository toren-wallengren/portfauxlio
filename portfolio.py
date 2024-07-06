from utils import generate_random_price_vectors, generate_random_unit_vectors, build_total_value_operator, \
    build_local_value_operator, build_loss_operator
import matplotlib.pyplot as plt
import numpy as np


class Portfolio:

    def __init__(self, num_of_assets, num_of_days, desired_portfolio_value):
        self.num_of_assets = num_of_assets
        self.num_of_days = num_of_days
        self.desired_portfolio_value = desired_portfolio_value

        # Generate random price vectors
        self.price_vectors = generate_random_price_vectors(self.num_of_assets, self.num_of_days)

        # Calculate total value vector
        self.total_value_vector = np.ones(self.num_of_days + 1) * self.desired_portfolio_value

        # Generate random unit vectors
        self.unit_vectors = generate_random_unit_vectors(self.price_vectors, self.total_value_vector)

    def update_total_value_vector(self):
        self.total_value_vector = np.sum(self.price_vectors*self.unit_vectors, axis=0)

    def perform_gradient_descent(self, learning_rate=0.1, iterations=100):
        T = self.num_of_days
        N = self.num_of_assets
        D = build_total_value_operator(T, self.desired_portfolio_value)
        DtD = D.T @ D

        initial_units = self.unit_vectors[:, 1]
        K = [build_local_value_operator(T, initial_units[n]) for n in range(N)]

        for _ in range(iterations):
            gradients = np.zeros_like(self.unit_vectors)
            total_value_norm = np.linalg.norm(self.total_value_vector)
            for n in range(N):
                # gradient for difference operator
                price_n = np.array(self.price_vectors[n])
                unit_n = self.unit_vectors[n, :]
                Kn = K[n]
                Ku = Kn @ unit_n
                Ku_norm = np.linalg.norm(Ku)
                gradients[n, :] = Kn.T @ Ku / Ku_norm

                for m in range(N):
                    unit_m = self.unit_vectors[m, :]
                    price_m = np.array(self.price_vectors[m])
                    val_m = price_m * unit_m
                    Qm = DtD @ val_m
                    term_m = price_n * Qm / total_value_norm
                    gradients[n, :] += term_m

                # Apply gradient descent update with stabilization
                gradients[int(n), 0] = 0
            self.unit_vectors -= learning_rate * gradients

            # Optional: Clamp values to avoid instability (depending on the specific problem constraints)
            self.unit_vectors = np.maximum(self.unit_vectors, 0)
            self.update_total_value_vector()

    def plot_unit_vectors(self):
        for i in range(self.unit_vectors.shape[0]):
            plt.plot(self.unit_vectors[i, :], label=f'Asset {i + 1}')
        plt.title('Unit Vectors')
        plt.xlabel('Days')
        plt.ylabel('Units')
        plt.legend()
        plt.show()

    def plot_price_vectors(self):
        for i in range(self.price_vectors.shape[0]):
            plt.plot(self.price_vectors[i, :], label=f'Asset {i + 1}')
        plt.title('Price Vectors')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def plot_value_vectors(self):
        for i in range(self.price_vectors.shape[0]):
            plt.plot(self.price_vectors[i, :]*self.unit_vectors[i, :], label=f'Asset {i + 1}')
        plt.title('Value Vectors')
        plt.xlabel('Days')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    def plot_total_value(self):
        total_value = np.sum(self.price_vectors*self.unit_vectors, axis=0)
        plt.plot(total_value)
        plt.title('Total Portfolio Value')
        plt.xlabel('Days')
        plt.ylabel('Value')
        plt.show()


    def plot_all(self):
        self.plot_unit_vectors()
        self.plot_price_vectors()
        self.plot_value_vectors()
        self.plot_total_value()