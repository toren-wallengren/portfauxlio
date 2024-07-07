from utils import generate_random_price_vectors, generate_random_unit_vectors, build_total_value_operator, \
    build_local_value_operator
import matplotlib.pyplot as plt
import numpy as np


class Portfolio:

    def __init__(self, price_vectors, desired_portfolio_value):
        self.price_vectors = price_vectors
        self.num_of_assets, self.num_of_days = price_vectors.shape
        self.desired_portfolio_value = desired_portfolio_value
        self.total_value_vector = desired_portfolio_value
        self.unit_vectors = generate_random_unit_vectors(self.price_vectors, self.total_value_vector)

    def update_total_value_vector(self):
        self.total_value_vector = np.sum(self.price_vectors * self.unit_vectors, axis=0)

    def perform_gradient_descent(self, learning_rate=0.01, iterations=100):
        T = self.num_of_days
        N = self.num_of_assets
        D = build_total_value_operator(T-1, self.desired_portfolio_value)
        DtD = D.T @ D

        initial_units = self.unit_vectors[:, 1]
        K = [build_local_value_operator(T-1, initial_units[n]) for n in range(N)]

        total_iterations = iterations * N * N
        current_iteration = 0
        display_threshold = 1

        for _ in range(iterations):
            gradients = np.zeros_like(self.unit_vectors)
            total_value_norm = np.linalg.norm(D @ self.total_value_vector)
            for n in range(N):
                # gradient for difference operator
                price_n = np.array(self.price_vectors[n])
                unit_n = self.unit_vectors[n, :]
                Kn = K[n]
                Ku = Kn @ unit_n
                Ku_norm = np.linalg.norm(Ku)
                gradients[n, :] = Kn.T @ Ku / Ku_norm

                for m in range(N):
                    current_iteration += 1
                    unit_m = self.unit_vectors[m, :]
                    price_m = np.array(self.price_vectors[m])
                    val_m = price_m * unit_m
                    Qm = DtD @ val_m
                    term_m = price_n * Qm / total_value_norm
                    gradients[n, :] += term_m

                # We don't want to change the first value (corresponds to a constant)
                gradients[int(n), 0] = 0
            self.unit_vectors -= learning_rate * gradients

            # Optional: Clamp values to avoid instability (depending on the specific problem constraints)
            self.unit_vectors = np.maximum(self.unit_vectors, 0)
            self.update_total_value_vector()

            percentage_complete = current_iteration / total_iterations * 100
            if percentage_complete >= display_threshold:
                print(f"Percentage complete: {percentage_complete}%")
                display_threshold += 1

    def plot_unit_vectors(self):
        for i in range(self.unit_vectors.shape[0]):
            plt.plot(self.unit_vectors[i, :], label=f'Asset {i + 1}')
        plt.title('Unit Vectors')
        plt.xlabel('Days')
        plt.ylabel('Units')
        #plt.legend()
        plt.show()

    def plot_price_vectors(self):
        for i in range(self.price_vectors.shape[0]):
            plt.plot(self.price_vectors[i, :], label=f'Asset {i + 1}')
        plt.title('Price Vectors')
        plt.xlabel('Days')
        plt.ylabel('Price')
        #plt.legend()
        plt.show()

    def plot_value_vectors(self):
        for i in range(self.price_vectors.shape[0]):
            plt.plot(self.price_vectors[i, :] * self.unit_vectors[i, :], label=f'Asset {i + 1}')
        plt.title('Value Vectors')
        plt.xlabel('Days')
        plt.ylabel('Value')
        #plt.legend()
        plt.show()

    def plot_total_value(self):
        total_value = np.sum(self.price_vectors * self.unit_vectors, axis=0)
        plt.plot(total_value)
        plt.title('Total Portfolio Value')
        plt.xlabel('Days')
        plt.ylabel('Value')
        plt.show()

    def plot_all(self):
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        # Plot unit vectors
        for i in range(self.unit_vectors.shape[0]):
            axs[0, 0].plot(self.unit_vectors[i, :], label=f'Asset {i + 1}')
        axs[0, 0].set_title('Unit Vectors')
        axs[0, 0].set_xlabel('Days')
        axs[0, 0].set_ylabel('Units')
        #axs[0, 0].legend()

        # Plot price vectors
        for i in range(self.price_vectors.shape[0]):
            axs[0, 1].plot(self.price_vectors[i, :], label=f'Asset {i + 1}')
        axs[0, 1].set_title('Price Vectors')
        axs[0, 1].set_xlabel('Days')
        axs[0, 1].set_ylabel('Price')
        #axs[0, 1].legend()

        # Plot value vectors
        for i in range(self.price_vectors.shape[0]):
            axs[1, 0].plot(self.price_vectors[i, :] * self.unit_vectors[i, :], label=f'Asset {i + 1}')
        axs[1, 0].set_title('Value Vectors')
        axs[1, 0].set_xlabel('Days')
        axs[1, 0].set_ylabel('Value')
        #axs[1, 0].legend()

        # Plot total value
        total_value = np.sum(self.price_vectors * self.unit_vectors, axis=0)
        axs[1, 1].plot(total_value)
        axs[1, 1].set_title('Total Portfolio Value')
        axs[1, 1].set_xlabel('Days')
        axs[1, 1].set_ylabel('Value')

        plt.tight_layout()
        plt.show()
