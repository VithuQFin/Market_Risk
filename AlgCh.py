# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 15:46:25 2024

@author: Kvith
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###############################################################################
# 1) READING AND PREPARING DATA
###############################################################################

def read_and_prepare_data(csv_path):
    """
      - 'time'   : temps
      - 'spread' : bid-ask spread
      - 'volume' : volume
      - 'sign'   : +1=vente (ou -1=achat, après inversion)
      - 'price'  : prix avant transaction
      - 'delta_S': variation (S_{k+1} - S_k)
    """
    df_raw = pd.read_csv(csv_path, sep=';', decimal=',', engine='python')

    df_raw.columns = ["time_raw", "spread_raw", "volume_raw", "sign_raw", "price_raw"]

    df_raw["time"]   = pd.to_numeric(df_raw["time_raw"],   errors='coerce')
    df_raw["spread"] = pd.to_numeric(df_raw["spread_raw"], errors='coerce')
    df_raw["volume"] = pd.to_numeric(df_raw["volume_raw"], errors='coerce').fillna(1)
    df_raw["sign"]   = pd.to_numeric(df_raw["sign_raw"],   errors='coerce')
    df_raw["price"]  = pd.to_numeric(df_raw["price_raw"],  errors='coerce')

    df_clean = df_raw.dropna(subset=["sign", "price"]).copy()

    # Convention: +1 = sale
    df_clean["sign"] = -1 * df_clean["sign"]

    # Chronological sorting
    df_clean.sort_values(by="time", inplace=True)

    # Calculation of delta_S
    df_clean["delta_S"] = df_clean["price"].shift(-1) - df_clean["price"]
    df_clean.dropna(subset=["delta_S"], inplace=True)

    return df_clean


###############################################################################
# 2) REGRESSION FUNCTION
###############################################################################

def simple_linear_regression(X, Y):
    """
    Performs linear regression Y = alpha + beta * X (one variable):
      - Calculates alpha, beta
      - Calculates residuals, MSE, R^2
    Returns :
      alpha, beta, (dict containing mse, r2, resid, etc.)
    """
    X = np.array(X)
    Y = np.array(Y)
    
    n = len(X)
    if n < 2:
        raise ValueError("Not enough points for linear regression.")
    
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)
    
    varX = np.sum((X - X_mean)**2)
    covXY = np.sum((X - X_mean)*(Y - Y_mean))
    
    if varX == 0:
        raise ValueError("Impossible to calculate slope (Var(X)=0).")
    
    beta = covXY / varX
    
    alpha = Y_mean - beta * X_mean
    
    Y_pred = alpha + beta * X
    resid = Y - Y_pred
    
    mse = np.mean(resid**2)
    
    sse = np.sum(resid**2)
    sst = np.sum((Y - Y_mean)**2)
    if sst > 0:
        r2 = 1 - sse/sst
    else:
        r2 = 1.0
    
    metrics = {"mse": mse, "r2": r2, "resid": resid, "alpha": alpha, "beta": beta}
    
    return alpha, beta, metrics


###############################################################################
# 3) ESTIMATION OF PERMANENT AND TRANSITORY IMPACT
###############################################################################

def estimate_permanent_impact_manual(df, tau=1.0):
    """
    Permanent impact : Delta S_k = alpha - gamma*n_k + sigma*sqrt(tau)*eps_k
    Assumption: n_k = volume * sign
    Regression (Delta S_k) ~ alpha + beta*(n_k),
      gamma_est = -beta.
    Then sigma_est = sqrt(Var(resid)/tau).
    Returns gamma_est, sigma_est, alpha_est, metrics (including r2, mse, etc.).
    """
    df["n_k"] = df["volume"] * df["sign"]
    X = df["n_k"]
    Y = df["delta_S"]
    
    alpha_est, beta_est, metrics = simple_linear_regression(X, Y)
    gamma_est = - beta_est
    
    resid = metrics["resid"]
    var_resid = np.var(resid, ddof=1)
    sigma_est = np.sqrt(var_resid / tau)
    
    return gamma_est, sigma_est, alpha_est, metrics


def estimate_transitory_impact_manual(df, tau=1.0):
    """
    Transient impact : S_k^+ - S_k = alpha_trans - [ xi*sgn(n_k) + eta*(n_k/tau) ]
    Approx : S_k^+ = S_k + sign(n_k)*spread/2 => delta_exec = sign(n_k)*(spread/2)
    Regress (S_k^+ - S_k) = alpha + b1*sgn(n_k) + b2*(n_k/tau)
      => b1=-xi, b2=-eta
    Returns xi_est, eta_est, alpha_trans, metrics
    """
    df["S_k_plus"] = df["price"] + df["sign"]*(df["spread"]/2)
    df["delta_exec"] = df["S_k_plus"] - df["price"]
    
    df["n_k"] = df["volume"] * df["sign"]
    df["sgn_nk"] = np.sign(df["n_k"])
    df["v_k"] = df["n_k"] / tau
    
    # Manual multiple regression on 2 variables
    #=> We can perform the regression in 2 steps, or pseudo-inversion of the matrix.
    # Here, to simplify, we'll just stack the matrix.
    
    # Matrix design : [ [1, sgn_nk, v_k], ... ]
    X_mat = np.column_stack((np.ones(len(df)), df["sgn_nk"], df["v_k"]))
    Y = df["delta_exec"].values
    
    # We want to solve : X_mat @ coeffs = Y en LS => coeffs = (X_mat^T X_mat)^(-1) X_mat^T Y
    # coeffs = [ alpha_trans, b1, b2]
    XtX = X_mat.T @ X_mat
    XtY = X_mat.T @ Y
    
    # Inversion
    coeffs = np.linalg.inv(XtX) @ XtY
    
    alpha_trans = coeffs[0]
    b1 = coeffs[1]
    b2 = coeffs[2]
    
    xi_est = -b1
    eta_est = -b2
    
    Y_pred = X_mat @ coeffs
    resid = Y - Y_pred
    
    mse = np.mean(resid**2)
    
    Y_mean = np.mean(Y)
    sse = np.sum(resid**2)
    sst = np.sum((Y - Y_mean)**2)
    r2 = 1 - sse/sst if sst>0 else 1.0
    
    metrics = {"alpha_trans": alpha_trans, "b1": b1, "b2": b2, "xi_est": xi_est, "eta_est": eta_est, "mse": mse, "r2": r2, "resid": resid}
    
    return xi_est, eta_est, alpha_trans, metrics


###############################################################################
# 4) STRATEGIES: CONTINUOUS AND DISCRETE (ALMGREN-CHRISS)
###############################################################################

def almgren_chriss_continuous_schedule(X0, T, sigma, gamma, lam, n_points=200):
    """
    Continuous Profil : x(t) = X0 * sinh[kappa*(T - t)] / sinh[kappa*T],
                     kappa = sqrt(lam * sigma^2 / gamma).
    """
    kappa = np.sqrt(lam * sigma**2 / gamma)
    times = np.linspace(0, T, n_points + 1)
    # Avoid division by zero if kappa=0 => We assume lam>0
    # Otherwise, x(t) ~ linear
    if np.isclose(kappa, 0.0):
        # Case lam=0 => run at ~constant speed
        x_opt = X0 * (1 - times/T)
    else:
        x_opt = X0 * np.sinh(kappa*(T - times)) / np.sinh(kappa*T)
    return times, x_opt

def almgren_chriss_discrete_schedule(X0, T, N, sigma, gamma, lam):
    """
    Discrete Profil : x_k = X0 * sinh[kappa*(N - k)*Delta]/sinh[kappa*N*Delta],
                     Delta = T/N.
    """
    Delta = T / N
    kappa = np.sqrt(lam * sigma**2 / gamma)
    times = np.array([k*Delta for k in range(N+1)])
    x_opt = np.zeros(N+1)
    
    if np.isclose(kappa, 0.0):
        # lam=0 => linear execution
        for k in range(N+1):
            x_opt[k] = X0*(1 - k/N)
    else:
        for k in range(N+1):
            numerator   = np.sinh(kappa * (N - k) * Delta)
            denominator = np.sinh(kappa * N * Delta)
            x_opt[k]    = X0*(numerator/denominator)
    
    return times, x_opt


###############################################################################
# 5) PLOT FUNCTION: DIFFERENT RISK AVERSIONS
###############################################################################

def plot_continuous_and_discrete_for_different_lambdas(
    X0, T_hours, N, sigma, gamma, lambdas
):
    plt.figure(figsize=(10, 6))
    
    for lam in lambdas:
        kappa = np.sqrt(lam * sigma**2 / gamma)
        print(f"[DEBUG] lambda={lam:.2e}, kappa={kappa:.4f}")
        
        t_cont, x_cont = almgren_chriss_continuous_schedule(X0, T_hours, sigma, gamma, lam)
        t_disc, x_disc = almgren_chriss_discrete_schedule(X0, T_hours, N, sigma, gamma, lam)
        
        plt.plot(t_cont, x_cont, label=f"Continuous (λ={lam:g})")
        plt.step(t_disc, x_disc, where='post', linestyle='--', 
                 label=f"Discrete (λ={lam:g}, N={N})")
    
    plt.title("Almgren-Chriss: Continuous vs. Discrete strategy for different λ")
    plt.xlabel("Time (hours)")
    plt.ylabel("Position x(t) (remaining shares)")
    plt.grid(True)
    plt.legend()
    plt.show()


###############################################################################
# 6) MAIN
###############################################################################

def main():
    csv_path = r"C:\Users\Kvith\OneDrive - De Vinci\Documents\ESILV 23-26\Cours A4\S1\Market Risk\Market Risk Project\DatasetTD4.csv"
    df_data  = read_and_prepare_data(csv_path)

    tau = 1.0
    gamma_est, sigma_est, alpha_perm, perm_metrics = estimate_permanent_impact_manual(df_data, tau)
    
    xi_est, eta_est, alpha_trans, trans_metrics = estimate_transitory_impact_manual(df_data, tau)
    
    print("=== Estimated parameters (Permanent impact) ===")
    print(f"alpha_perm   = {alpha_perm:.6f}")
    print(f"gamma        = {gamma_est:.6f}")
    print(f"sigma        = {sigma_est:.6f}")
    print(f"--> MSE permanent = {perm_metrics['mse']:.6e}, R^2={perm_metrics['r2']:.4f}\n")
    
    print("=== Estimated parameters (transitory impact) ===")
    print(f"alpha_trans  = {alpha_trans:.6f}")
    print(f"xi           = {xi_est:.6f}")
    print(f"eta          = {eta_est:.6f}")
    print(f"--> MSE transitory = {trans_metrics['mse']:.6e}, R^2={trans_metrics['r2']:.4f}\n")
    
    X0      = 1_000_000  # 1 million shares
    T_hours = 72.0       # 72 h
    N       = 72
    lambdas = [1e-6, 1e-4, 1e-2]
    
    plot_continuous_and_discrete_for_different_lambdas(X0, T_hours, N, sigma_est, gamma_est, lambdas)

if __name__ == "__main__":
    main()
