import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import time
import sys



def get_data(tickers, start, end):
    """
    It downloads historical adjusted closing prices for a list of stock tickers from Yahoo Finance.
    After it downloaded checks if data valid or not , or there is/are error in data.
    Args:
        tickers(list): A list of stock ticker symbols (e.g., ['AAPL', 'MSFT']).
        start (str or datetime): The start date for the data retrieval.
        end (str or datetime): The end date for the data retrieval.
    Returns:
              pd.DataFrame: A pandas DataFrame containing the adjusted closing prices,
                      with dates as the index and tickers as the columns.
                      Returns an empty DataFrame if data cannot be downloaded.

             Example structure:
                                  AAPL        MSFT        GOOG
                      Date
                      2021-01-04  128.519324  215.627472  1728.239990
                      2021-01-05  130.103012  215.845612  1740.920044
                      ...            ...         ...         ...
    """
    try:
        data = yf.download(tickers, start=start, end=end,auto_adjust=True)["Close"]
        if isinstance(data,pd.Series):
            data = data.to_frame(name=tickers[0])
        if data.empty:
            print(f"ERROR: No data found for tickers {tickers}.")
            return pd.DataFrame()
        if data.isnull().all().all():
            print(f"ERROR: All tickers provided {tickers} were invalid or had no data in selected range")
            return pd.DataFrame()
        return data
    except Exception as e:
        print(f"AN UNEXPECTED ERROR OCCURED: {e}")
        return pd.DataFrame()


def calculate_statistics(data):
    """
    Calculates the annualized mean returns and covariance matrix from price data.
    Args:
        data (pd.DataFrame): It takes the data from get_data() function.This is historical asset adjusted with close price.

    Returns:
          tuple: A tuple containing:
            - pd.Series: Annualized mean of logarithmic returns.
            - pd.DataFrame: Annualized covariance matrix of logarithmic returns.
    """
    daily_returns = data.pct_change().dropna()
    annualized_mean_returns = daily_returns.mean() *252
    annualized_cov_matrix = daily_returns.cov() *252
    return annualized_mean_returns, annualized_cov_matrix


def run_monte_carlo_simulation(mean_returns, cov_matrix, number_of_simulations,progress_bar = None):
    """
        Performs a Monte Carlo simulation to generate random portfolio allocations.

    Args:
        mean_returns (pd.Series): Annualized mean returns of assets.
        cov_matrix (pd.DataFrame): Annualized covariance matrix of assets.
        number_of_simulations (int): The number of random portfolios to generate.

    Returns:
        pd.DataFrame: A DataFrame containing the returns, risks, and Sharpe Ratios
                      for each simulated portfolio.
    """
    portfolio_return = []
    portfolio_risk = []
    sharpe_ratios = []

    start_time = time.time()
    for i in range(number_of_simulations):
        weights = np.random.rand(len(mean_returns))
        weights = weights / np.sum(weights)

        total_return = np.sum(mean_returns * weights)
        portfolio_return.append(total_return)

        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        portfolio_risk.append(portfolio_volatility)

        sharpe_ratio = total_return / portfolio_volatility
        sharpe_ratios.append(sharpe_ratio)

        if progress_bar and (i + 1) % 10000 == 0:
            progress = (i + 1) / number_of_simulations
            elapsed_time = time.time() - start_time

            sims_left = number_of_simulations - (i + 1)
            time_per_sim = elapsed_time / (i + 1)
            eta_seconds = sims_left * time_per_sim

            if eta_seconds > 60:
                eta_minutes = int(eta_seconds // 60)
                eta_rem_seconds = int(eta_seconds % 60)
                eta_str = f"{eta_minutes}m {eta_rem_seconds}s"
            else:
                eta_str = f"{int(eta_seconds)}s"

            progress_text = f"Running... {int(progress * 100)}% complete. (ETA: {eta_str})"
            progress_bar.progress(progress, text=progress_text)
            time.sleep(0.001)
    results_dict = {
        "Return": portfolio_return,
        "Risk": portfolio_risk,
        "Sharpe Ratio": sharpe_ratios
    }
    if progress_bar:
        progress_bar.progress(1.0, text="Simulation Complete!")
    results_df = pd.DataFrame(results_dict)
    return results_df


def negative_sharpe_ratio(weights, mean_returns, cov_matrix):
    """
    This part is used by SciPy optimizer, which minimizes a function. Since there is no method of
    maximization in SciPy, it better to find min point o the function(mathematical) and then reverse it the point.
    This strategic approach gives the maximum point of the function(mathematical).This function calculates
    the negative Sharpe Ratio for a given set of weights.

    Args:
        weights (np.array): Portfolio weights.
        mean_returns (pd.Series): Annualized mean returns.
        cov_matrix (pd.DataFrame): Annualized covariance matrix.

    Returns:
        float: The negative Sharpe Ratio.
    """
    total_return = np.sum(mean_returns * weights)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    if portfolio_volatility == 0:
        return 0
    sharpe_ratio = total_return / portfolio_volatility
    return -sharpe_ratio


def find_optimal_portfolio(mean_returns, cov_matrix):
    """
     Here this function uses mathematical optimization to find the optimal portfolio weights. The special
     method is name as 'SLSQP', Sequential Least Squares Programming, good for constrained optimization.
     It create random weights until find a weight bundle that sums equal to 1. This must be because each weight
     represents ratio of a person total money  who invest his/her money into bundle of a stocks.

     Args:
        mean_returns (pd.Series): Annualized mean returns.
        cov_matrix (pd.DataFrame): Annualized covariance matrix.

     Returns:
          np.array: An array of optimal portfolio weights.
    """
    asset_length = len(mean_returns)
    bounds = asset_length * ((0, 1),)
    constraints = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }
    initial_guess = asset_length * [1 / asset_length, ]
    solution = sco.minimize(
        fun=negative_sharpe_ratio,
        x0=initial_guess,
        args=(mean_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return solution.x


if __name__ == "__main__":
    # This block runs only when the script is executed directly (for testing purposes)
    print("--- Running Initial Tests ---")
    test_tickers = ['SPY', 'TLT']
    test_start = '2019-01-01'
    test_end = '2021-12-31'
    print(f"\nAttempting to download data for {test_tickers}...")
    price_data = get_data(test_tickers, test_start, test_end)
    if not price_data.empty:
        print("\nData download successful. Now calculating statistics...")
        mean_returns, cov_matrix= calculate_statistics(price_data)
        print("\nCalculated Annualized Mean Returns:")
        print(mean_returns)
        print("\nCalculated Annualized Covariance Matrix:")
        print(cov_matrix)
        print("\nRunning Monte Carlo simulation...")
        simulation_result = run_monte_carlo_simulation(mean_returns, cov_matrix, 10_000)
        print("\nSimulation Results:")
        print(simulation_result.head())
        max_sharpe_idx = simulation_result["Sharpe Ratio"].idxmax()
        best_portfolio = simulation_result.iloc[max_sharpe_idx]
        print("\nBest portfolio:")
        print(best_portfolio)
        print("\n--- Running SciPy Optimizer ---")
        optimal_weights = find_optimal_portfolio(mean_returns, cov_matrix)
        optimal_weights_series = pd.Series(optimal_weights, index=mean_returns.index)
        print(optimal_weights_series)
        optimal_return = np.sum(mean_returns * optimal_weights)
        optimal_variance = np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
        optimal_volatility = np.sqrt(optimal_variance)
        optimal_sharpe = optimal_return / optimal_volatility
        print("\nOptimal Portfolio Performance (from SciPy Optimizer):")
        print(f"Return: {optimal_return:.4f}")
        print(f"Volatility: {optimal_volatility:.4f}")
        print(f"Sharpe Ratio: {optimal_sharpe:.4f}")

    else:
        print("\nSkipping statistics calculation because data download failed.")
    print("\n--- Initial Tests Finished ---")


 # faktör modelleri !! bak
