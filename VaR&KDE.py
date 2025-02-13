import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

###############################################################################
# 1) READING AND PREPARING DATA
###############################################################################

def read_and_prepare_data(csv_path):
    """
    Reads the dataset, formats it, and computes daily returns.

    Returns:
        df_clean (DataFrame): Cleaned dataset with computed returns.
    """
    df = pd.read_csv(csv_path, sep=';', header=0, decimal=',')
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['Close'] = df['Close'].astype(str).str.replace(',', '.').astype(float)

    # Compute daily returns
    df['rendements'] = df['Close'].pct_change()
    df.dropna(inplace=True)  # Remove first NaN due to pct_change()

    return df


###############################################################################
# 2) CALCULATE NON-PARAMETRIC VAR
###############################################################################

def empirical_cdf(x, data):
    """ Compute empirical CDF for a given value x. """
    return np.sum(data <= x) / len(data)

def compute_var(returns, alpha=0.95):
    """
    Computes the non-parametric Value-at-Risk (VaR) using the empirical CDF.

    Returns:
        non_parametric_var (float): Estimated VaR.
    """
    x_range = np.linspace(min(returns), max(returns), 100000)
    cdf_values = [empirical_cdf(x, returns) for x in x_range]

    for i, cdf in enumerate(cdf_values):
        if cdf >= (1 - alpha):
            return x_range[i]
    return None


###############################################################################
# 3) KERNEL DENSITY ESTIMATION
###############################################################################

def logistic_kernel(u):
    """ Logistic Kernel function. """
    exp_neg_u = np.exp(-u)
    return exp_neg_u / (1 + exp_neg_u) ** 2

def kernel_density_estimate(x_range, data, h):
    """ Kernel density estimation using the logistic kernel. """
    n = len(data)
    kde_values = [np.sum(logistic_kernel((x - data) / h)) / (n * h) for x in x_range]
    return kde_values

def estimate_kde(returns):
    """
    Estimate kernel density for different bandwidths.
    
    Returns:
        x_range (array): Range of values for KDE computation.
        kde_values (dict): Dictionary with KDE values for different bandwidths.
    """
    x_range = np.linspace(min(returns), max(returns), 100000)
    n = len(returns)
    
    # Different bandwidths
    h1 = 1.06 * np.std(returns) * (n ** (-1 / 5))
    h2 = 0.17 * np.std(returns) * (n ** (-1 / 5))
    h3 = 3.04 * np.std(returns) * (n ** (-1 / 5))

    kde_values = {
        'h1': kernel_density_estimate(x_range, returns, h1),
        'h2': kernel_density_estimate(x_range, returns, h2),
        'h3': kernel_density_estimate(x_range, returns, h3)
    }

    return x_range, kde_values


###############################################################################
# 4) PLOTTING FUNCTION
###############################################################################

def plot_results(x_range, kde_values, cdf_values, var_threshold):
    """
    Plots Kernel Density Estimation (KDE) and empirical CDF with VaR.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(x_range, kde_values['h1'], label='Density (Logistic Kernel)', color='blue')
    plt.plot(x_range, kde_values['h2'], label='h small', color='lightblue')
    plt.plot(x_range, kde_values['h3'], label='h large', color='cyan')

    plt.plot(x_range, cdf_values, label='Empirical CDF', color='green')
    plt.axvline(var_threshold, color='red', linestyle='-', label='VaR')
    plt.xlim(-0.075, 0.078)
    plt.axvline(0, color='black')
    plt.axhline(0, color='black')
    
    plt.title("Density of Returns using Kernel Approaches & Empirical CDF")
    plt.ylabel("Density")
    plt.xlabel("Returns")
    plt.legend(loc='upper right', fontsize='small')
    plt.grid()
    plt.show()


###############################################################################
# 5) MAIN EXECUTION
###############################################################################

def main():
    csv_path = r"C:\Users\Kvith\OneDrive - De Vinci\Documents\GItHub\Market_Project_KAILASAPILLAI_HEBRARD\Market_Project_KAILASAPILLAI_HEBRARD\DATATD1.csv"  # Update with actual file path
    data = read_and_prepare_data(csv_path)

    # First dataset for VaR estimation
    data_set1 = data[(data['Date'] >= '2015-01-01') & (data['Date'] <= '2016-12-30')].dropna()
    returns = data_set1['rendements'].values

    var_threshold = compute_var(returns, alpha=0.95)
    print(f"Non-parametric VaR (95% confidence): {var_threshold}")

    # KDE computation
    x_range, kde_values = estimate_kde(returns)
    cdf_values = [empirical_cdf(x, returns) for x in x_range]

    # Plot results
    plot_results(x_range, kde_values, cdf_values, var_threshold)

    # Second dataset for VaR validation
    data_set2 = data[(data['Date'] >= '2017-01-01') & (data['Date'] <= '2018-12-30')].dropna()
    returns2 = data_set2['rendements'].values

    # Validation of the previously computed VaR
    VaR_95 = -0.03893
    exceed_percentage = np.mean(returns2 < VaR_95) * 100
    print(f"The percentage of values exceeding the VaR is {round(exceed_percentage, 3)}%, "
          f"which is {'<=' if exceed_percentage <= 5 else '>'} {round((1 - 0.95) * 100, 2)}%.")


if __name__ == "__main__":
    main()
