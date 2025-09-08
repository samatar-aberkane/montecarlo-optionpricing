# fonctions Monte Carlo

import numpy as np
from scipy.stats import norm

# -------------------------
# Black-Scholes formulas
# -------------------------
def black_scholes_call(S0, K, T, r, sigma):
    """Price a European call option using Black-Scholes formula."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def black_scholes_put(S0, K, T, r, sigma):
    """Price a European put option using Black-Scholes formula."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

# -------------------------
# Monte Carlo simulation
# -------------------------
def simulate_gbm(S0, T, r, sigma, N, M, antithetic=False, seed=None):
    """
    Simulate Geometric Brownian Motion
    S0: initial stock price
    T: time to maturity
    r: risk-free rate
    sigma: volatility
    N: number of time steps
    M: number of simulations
    antithetic: use antithetic variates
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / N
    if antithetic:
        half_M = M // 2
        Z = np.random.normal(0, 1, (half_M, N))
        Z = np.vstack([Z, -Z])
    else:
        Z = np.random.normal(0, 1, (M, N))
    
    S = np.zeros((M, N+1))
    S[:,0] = S0
    for t in range(1, N+1):
        S[:,t] = S[:,t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:,t-1])
    return S

# -------------------------
# European option pricing
# -------------------------
def european_call_mc(S0, K, T, r, sigma, N=100, M=10000, antithetic=False):
    """Monte Carlo pricing of a European call"""
    S = simulate_gbm(S0, T, r, sigma, N, M, antithetic)
    payoff = np.maximum(S[:,-1] - K, 0)
    return np.exp(-r*T) * np.mean(payoff)

def european_put_mc(S0, K, T, r, sigma, N=100, M=10000, antithetic=False):
    """Monte Carlo pricing of a European put"""
    S = simulate_gbm(S0, T, r, sigma, N, M, antithetic)
    payoff = np.maximum(K - S[:,-1], 0)
    return np.exp(-r*T) * np.mean(payoff)

# -------------------------
# Greeks (finite difference)
# -------------------------
def delta_mc(option_type, S0, K, T, r, sigma, eps=1e-4, **kwargs):
    """Approximate Delta via finite differences"""
    if option_type.lower() == 'call':
        price_up = european_call_mc(S0 + eps, K, T, r, sigma, **kwargs)
        price_down = european_call_mc(S0 - eps, K, T, r, sigma, **kwargs)
    elif option_type.lower() == 'put':
        price_up = european_put_mc(S0 + eps, K, T, r, sigma, **kwargs)
        price_down = european_put_mc(S0 - eps, K, T, r, sigma, **kwargs)
    else:
        raise ValueError("Option type must be 'call' or 'put'")
    return (price_up - price_down) / (2 * eps)

def vega_mc(option_type, S0, K, T, r, sigma, eps=1e-4, **kwargs):
    """Approximate Vega via finite differences"""
    price_up = european_call_mc(S0, K, T, r, sigma + eps, **kwargs) if option_type.lower() == 'call' else european_put_mc(S0, K, T, r, sigma + eps, **kwargs)
    price_down = european_call_mc(S0, K, T, r, sigma - eps, **kwargs) if option_type.lower() == 'call' else european_put_mc(S0, K, T, r, sigma - eps, **kwargs)
    return (price_up - price_down) / (2 * eps)

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    S0 = 100
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2

    print("European Call (MC) =", european_call_mc(S0, K, T, r, sigma, M=50000, antithetic=True))
    print("European Call (BS) =", black_scholes_call(S0, K, T, r, sigma))
    print("Delta Call (MC) =", delta_mc('call', S0, K, T, r, sigma, M=50000, antithetic=True))
    print("Vega Call (MC) =", vega_mc('call', S0, K, T, r, sigma, M=50000, antithetic=True))
