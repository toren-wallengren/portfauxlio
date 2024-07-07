from portfolio import Portfolio
from data_generator import DataGenerator

num_of_assets = 5
num_of_days = 5

dg = DataGenerator()


price_vectors = dg.generate_random_price_vectors(num_of_assets, num_of_days)
desired_portfolio_value = dg.generate_random_portfolio_value_vector(num_of_days, distribution='normal', sigma=100)
portfolio = Portfolio(price_vectors, desired_portfolio_value)

portfolio.perform_gradient_descent()