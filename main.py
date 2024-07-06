from portfolio import Portfolio

num_of_assets = 5
num_of_days = 5
desired_portfolio_value = 1000

portfolio = Portfolio(num_of_assets, num_of_days, desired_portfolio_value)

portfolio.perform_gradient_descent()