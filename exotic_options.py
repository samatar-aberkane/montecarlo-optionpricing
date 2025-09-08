# barrier / Asian

import numpy as np
from pricing import simulate_gbm

# -------------------------
# Asian Option (Arithmetic Average)
# -------------------------
def asian_call_mc(S0, K, T, r, sigma, N=100, M=10000, antithetic=False, seed=None):
    """
    Monte Carlo pricing of an Asian call option (arithmetic average)
    """
    S = simulate_gbm(S0, T, r, sigma, N, M, antithetic, seed)
    # Average over all time steps (excluding S0)
    avg_price = np.mean(S[:, 1:], axis=1)
    payoff = np.maximum(avg_price - K, 0)
    return np.exp(-r*T) * np.mean(payoff)

def asian_put_mc(S0, K, T, r, sigma, N=100, M=10000, antithetic=False, seed=None):
    """
    Monte Carlo pricing of an Asian put option (arithmetic average)
    """
    S = simulate_gbm(S0, T, r, sigma, N, M, antithetic, seed)
    avg_price = np.mean(S[:, 1:], axis=1)
    payoff = np.maximum(K - avg_price, 0)
    return np.exp(-r*T) * np.mean(payoff)

# -------------------------
# Barrier Option (Up-and-Out Call)
# -------------------------
def barrier_call_mc(S0, K, B, T, r, sigma, N=100, M=10000, antithetic=False, seed=None):
    """
    Monte Carlo pricing of an up-and-out call option
    B: barrier level
    """
    S = simulate_gbm(S0, T, r, sigma, N, M, antithetic, seed)
    # Check if barrier was breached
    breached = np.any(S[:,1:] >= B, axis=1)
    payoff = np.where(breached, 0, np.maximum(S[:,-1] - K, 0))
    return np.exp(-r*T) * np.mean(payoff)

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    S0 = 100
    K = 100
    B = 120
    T = 1.0
    r = 0.05
    sigma = 0.2

    print("Asian Call (MC) =", asian_call_mc(S0, K, T, r, sigma, M=50000, antithetic=True))
    print("Asian Put (MC) =", asian_put_mc(S0, K, T, r, sigma, M=50000, antithetic=True))
    print("Barrier Call Up-and-Out (MC) =", barrier_call_mc(S0, K, B, T, r, sigma, M=50000, antithetic=True))
