# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 16:52:49 2024

@author: Kvith
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 14:32:30 2024

@author: Kvith
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

###############################################################################
# 1) DATA LOADING AND CLEANING
###############################################################################

def load_and_clean_data(file_path):
    """
    Load and clean data from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        data = pd.read_csv(file_path, sep=';')
    except pd.errors.ParserError:
        try:
            data = pd.read_csv(file_path, delimiter=',', decimal=',')
        except pd.errors.ParserError:
            data = pd.read_csv(file_path, delimiter='\t', decimal=',')

    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y', errors='coerce')

    if 'prix' in data.columns:
        data['prix'] = data['prix'].str.replace(',', '.').astype(float, errors='ignore')

    if 'rendements' not in data.columns or data['rendements'].isnull().any():
        data['rendements'] = data['prix'].pct_change()

    data.dropna(inplace=True)

    return data

###############################################################################
# 2) BLACK-SCHOLES MODEL
###############################################################################

def black_scholes_call_price(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes price of a call option.

    Parameters:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to maturity (in years).
        r (float): Risk-free interest rate.
        sigma (float): Volatility of the stock.

    Returns:
        float: Price of the call option.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

###############################################################################
# 3) SIMULATION AND VAR
###############################################################################

def simulate_prices_and_calculate_var(S0, mu, sigma, T, K, r, alpha=0.05, N=5000):
    """
    Simulate stock prices and calculate Value at Risk (VaR) for a call option.

    Parameters:
        S0 (float): Initial stock price.
        mu (float): Expected return.
        sigma (float): Volatility.
        T (float): Time to maturity (in years).
        K (float): Strike price.
        r (float): Risk-free interest rate.
        alpha (float): Confidence level for VaR calculation.
        N (int): Number of Monte Carlo simulations.

    Returns:
        float: VaR for the call option.
    """
    np.random.seed(42)
    simulated_prices = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.normal(0, 1, N))
    call_prices = np.array([black_scholes_call_price(S, K, T, r, sigma) for S in simulated_prices])
    var_call = np.percentile(call_prices, alpha * 100)
    return call_prices, var_call

###############################################################################
# 4) VISUALIZATION
###############################################################################

def plot_results(data, simulated_prices, call_prices, var_call):
    """
    Plot histograms of returns, simulated prices, and call prices.

    Parameters:
        data (pd.DataFrame): Original data.
        simulated_prices (np.ndarray): Simulated stock prices.
        call_prices (np.ndarray): Simulated call prices.
        var_call (float): Value at Risk for the call option.
    """
    plt.figure(figsize=(12, 6))

    # Distribution of returns
    plt.subplot(2, 2, 1)
    plt.hist(data['rendements'], bins=30, alpha=0.7, color='blue')
    plt.title("Returns distribution")
    plt.xlabel("Returns")
    plt.ylabel("Frequency")

    # Distribution of simulated prices
    plt.subplot(2, 2, 2)
    plt.hist(simulated_prices, bins=30, alpha=0.7, color='green')
    plt.title("Distribution of simulated prices")
    plt.xlabel("Simulated prices")
    plt.ylabel("Frequency")

    # Distribution of call prices
    plt.subplot(2, 2, 3)
    plt.hist(call_prices, bins=30, alpha=0.7, color='orange')
    plt.title("Distribution of simulated call prices")
    plt.xlabel("Call prices")
    plt.ylabel("Frequency")

    # VaR plot
    plt.subplot(2, 2, 4)
    plt.axvline(var_call, color='red', linestyle='dashed', linewidth=2, label=f"VaR 5%: {var_call:.4f}")
    plt.hist(call_prices, bins=30, alpha=0.7, color='purple')
    plt.title("VaR of call prices")
    plt.xlabel("Call prices")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.show()

###############################################################################
# 5) MAIN FUNCTION
###############################################################################

def main():
    file_path = r"C:\Users\Kvith\OneDrive - De Vinci\Documents\ESILV 23-26\Cours A4\S1\Market Risk\Market Risk Project\DATATD12.csv"
    data = load_and_clean_data(file_path)

    S0 = data['prix'].iloc[-1]  # Last observed price
    K = S0  # At-the-money option
    T = 1 / 252  # 1-day maturity
    r = 0.0422  # Risk-free rate

    # Weighted parameters for returns and volatility
    lambda_ = 0.94
    weights = np.exp(-lambda_ * np.arange(len(data['rendements'])))
    weights /= weights.sum()

    mu = np.sum(weights * data['rendements'])
    sigma = np.sqrt(np.sum(weights * (data['rendements'] - mu)**2))

    call_prices, var_call = simulate_prices_and_calculate_var(S0, mu, sigma, T, K, r)

    print("=== Parameters ===")
    print(f"mu = {mu:.6f}")
    print(f"sigma = {sigma:.6f}")
    print("=== Results ===")
    print(f"VaR (5%): {var_call:.6f}")

    plot_results(data, S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.normal(0, 1, 5000)), call_prices, var_call)

if __name__ == "__main__":
    main()
