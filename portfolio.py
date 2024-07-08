import matplotlib.pyplot as plt
import numpy as np


class Portfolio:

    def __init__(self, unit_vectors, price_vectors, desired_portfolio_value, objective_functions):
        self.unit_vectors = unit_vectors
        self.price_vectors = price_vectors
        self.num_of_assets, self.num_of_days = price_vectors.shape
        self.desired_portfolio_value = desired_portfolio_value
        self.objective_functions = objective_functions

    def perform_gradient_descent(self, learning_rate=0.001, iterations=100):
        total_iterations = iterations
        current_iteration = 0
        display_threshold = 1

        for _ in range(iterations):
            gradients_total = np.zeros_like(self.unit_vectors)
            for objective_function in self.objective_functions:
                gradients_total += objective_function.compute_gradient(self.unit_vectors)
            gradients_total[:, 0] = 0
            gradients_total[:, 1] = 0
            self.unit_vectors -= learning_rate * gradients_total
            percentage_complete = current_iteration / total_iterations * 100
            current_iteration += 1
            if percentage_complete >= display_threshold:
                print(f"Percentage complete: {percentage_complete}%")
                display_threshold += 1

    def plot_unit_vectors(self):
        for i in range(self.unit_vectors.shape[0]):
            plt.plot(self.unit_vectors[i, :], label=f'Asset {i + 1}')
        plt.title('Unit Vectors')
        plt.xlabel('Days')
        plt.ylabel('Units')
        # plt.legend()
        plt.show()

    def plot_price_vectors(self):
        for i in range(self.price_vectors.shape[0]):
            plt.plot(self.price_vectors[i, :], label=f'Asset {i + 1}')
        plt.title('Price Vectors')
        plt.xlabel('Days')
        plt.ylabel('Price')
        # plt.legend()
        plt.show()

    def plot_value_vectors(self):
        for i in range(self.price_vectors.shape[0]):
            plt.plot(self.price_vectors[i, :] * self.unit_vectors[i, :], label=f'Asset {i + 1}')
        plt.title('Value Vectors')
        plt.xlabel('Days')
        plt.ylabel('Value')
        # plt.legend()
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
        # axs[0, 0].legend()

        # Plot price vectors
        for i in range(self.price_vectors.shape[0]):
            axs[0, 1].plot(self.price_vectors[i, :], label=f'Asset {i + 1}')
        axs[0, 1].set_title('Price Vectors')
        axs[0, 1].set_xlabel('Days')
        axs[0, 1].set_ylabel('Price')
        # axs[0, 1].legend()

        # Plot value vectors
        for i in range(self.price_vectors.shape[0]):
            axs[1, 0].plot(self.price_vectors[i, :] * self.unit_vectors[i, :], label=f'Asset {i + 1}')
        axs[1, 0].set_title('Value Vectors')
        axs[1, 0].set_xlabel('Days')
        axs[1, 0].set_ylabel('Value')
        # axs[1, 0].legend()

        # Plot total value
        total_value = np.sum(self.price_vectors * self.unit_vectors, axis=0)
        axs[1, 1].plot(total_value)
        axs[1, 1].set_title('Total Portfolio Value')
        axs[1, 1].set_xlabel('Days')
        axs[1, 1].set_ylabel('Value')

        plt.tight_layout()
        plt.show()
