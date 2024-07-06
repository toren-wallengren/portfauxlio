import numpy as np


def initialize_portfolio(T, desired_portfolio_start_value, desired_portfolio_end_value, price_vectors):
    N = len(price_vectors)

    # Randomly distribute the desired portfolio value among the financial instruments
    random_weights = np.random.rand(N)
    random_weights /= random_weights.sum()

    # Calculate initial units based on the first day's prices
    initial_values = random_weights * desired_portfolio_start_value
    initial_units = [initial_values[i] / price_vectors[i][0] for i in range(N)]

    # Calculate final units based on the last day's prices
    # Randomly distribute the desired portfolio value among the financial instruments
    random_weights = np.random.rand(N)
    random_weights /= random_weights.sum()
    final_values = random_weights * desired_portfolio_end_value
    final_units = [final_values[i] / price_vectors[i][-1] for i in range(N)]

    # Create unit vectors that interpolate between initial and final units
    unit_vectors = np.zeros((N, T))
    for i in range(N):
        unit_vectors[i, :] = np.linspace(initial_units[i], final_units[i], T)

    return unit_vectors


def build_operator_D(T):
    D = np.zeros((T, T))
    for i in range(T):
        if i > 0:
            D[i, i - 1] = -1
        D[i, i] = 1
    return D


def calculate_portfolio_value(unit_vectors, price_vectors):
    N, T = unit_vectors.shape
    V = np.zeros(T)
    for t in range(T):
        for n in range(N):
            V[t] += unit_vectors[n, t] * price_vectors[n][t]
    return V


def gradient_descent(T, N, unit_vectors, price_vectors, learning_rate=0.00001, iterations=10000):
    D = build_operator_D(T)
    DtD = D.T @ D
    #DtD[0, 0] = 1

    for _ in range(iterations):
        gradients = np.zeros_like(unit_vectors)
        for n in range(N):
            price_n = np.array(price_vectors[n])
            for m in range(N):
                price_m = np.array(price_vectors[m])
                unit_m = unit_vectors[m, :]
                val_m = price_m * unit_m
                Qm = DtD @ val_m
                term_m = price_n * Qm
                gradients[n, :] += term_m

            # Apply gradient descent update with stabilization
            gradients[int(n), 0] = 0
            gradients[int(n), -1] = 0
        unit_vectors -= 2 * learning_rate * gradients

        # Optional: Clamp values to avoid instability (depending on the specific problem constraints)
        unit_vectors = np.maximum(unit_vectors, 0)

    return unit_vectors

def display_total_portfolio_value(unit_vectors, price_vectors):
    V = calculate_portfolio_value(unit_vectors, price_vectors)
    print("Total Portfolio Value:")
    print(V)

def generate_random_price_vectors(T, num_vectors, start_price_range=(10, 50), price_step_range=(0, 5)):
    """
    Generate random price vectors with a pattern similar to the provided example.

    Parameters:
    T (int): Number of days (length of each price vector).
    num_vectors (int): Number of price vectors to generate.
    start_price_range (tuple): Range of starting prices (min, max).
    price_step_range (tuple): Range of price step changes (min, max).

    Returns:
    list of list of float: Randomly generated price vectors.
    """
    price_vectors = []
    for _ in range(num_vectors):
        start_price = np.random.uniform(start_price_range[0], start_price_range[1])
        price_vector = [start_price]
        for _ in range(1, T):
            price_step = np.random.uniform(price_step_range[0], price_step_range[1])
            price_change = np.random.choice([-1, 1]) * price_step
            new_price = price_vector[-1] + price_change
            price_vector.append(new_price)
        price_vectors.append(price_vector)
    return price_vectors

# Example usage
T = 30
num_vectors = 5
price_vectors = generate_random_price_vectors(T, num_vectors)

print("Random Price Vectors:")
print(price_vectors)

unit_vectors = initialize_portfolio(T, 10000, 15000, price_vectors)
print("Initial Unit Vectors:")
print(unit_vectors.astype(int))

unit_vectors = gradient_descent(T, len(price_vectors), unit_vectors, price_vectors)

print("Optimized Unit Vectors:")
print(unit_vectors.astype(int))

display_total_portfolio_value(unit_vectors.astype(int), price_vectors)