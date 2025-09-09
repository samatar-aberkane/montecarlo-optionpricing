"""
Monte Carlo Option Pricing
"""

import numpy as np
from scipy.stats import norm
from typing import Optional, Tuple, Union
import warnings

class MCPricingEngine:
    """
    Monte Carlo pricing engine with advanced variance reduction
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize pricing engine with random generator"""
        self.rng = np.random.default_rng(seed)

    def simulate_gbm_vectorized(self, 
                               S0: float, 
                               T: float, 
                               r: float, 
                               sigma: float, 
                               N: int, 
                               M: int, 
                               antithetic: bool = False) -> np.ndarray:
        """
        Vectorized Geometric Brownian Motion simulation
        
        Parameters:
        -----------
        S0: Initial stock price
        T: Time to maturity
        r: Risk-free rate
        sigma: Volatility
        N: Number of time steps
        M: Number of simulations
        antithetic: Use antithetic variates
        
        Returns:
        --------
        S: Array of shape (M, N+1) containing price paths
        """
        dt = T / N
        
        # Adjust M for antithetic variates
        if antithetic and M % 2 != 0:
            warnings.warn("M must be even for antithetic variates. Using M+1 simulations.", UserWarning)
            M += 1

        # Generate all random numbers at once
        if antithetic:
            half_M = M // 2
            Z = self.rng.normal(0, 1, (half_M, N))
            Z = np.vstack([Z, -Z])
        else:
            Z = self.rng.normal(0, 1, (M, N))

        # Calculate log returns
        log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        
        # Calculate price paths
        log_S = np.zeros((M, N+1))
        log_S[:, 0] = np.log(S0)
        log_S[:, 1:] = np.log(S0) + np.cumsum(log_returns, axis=1)
        
        return np.exp(log_S)
    
    
    def black_scholes_call(self, S0: float, K: float, T: float, r: float, sigma: float) -> float:
        """Black-Scholes call option price"""
        if T <= 0:
            return max(S0 - K, 0)
        
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    
    def black_scholes_put(self, S0: float, K: float, T: float, r: float, sigma: float) -> float:
        """Black-Scholes put option price"""
        if T <= 0:
            return max(K - S0, 0)
            
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r*T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    
    def european_call_mc(self, 
                        S0: float, 
                        K: float, 
                        T: float, 
                        r: float, 
                        sigma: float,
                        N: int = 100, 
                        M: int = 100000, 
                        antithetic: bool = True,
                        control_variate: bool = False) -> Union[float, Tuple[float, float]]:
        """
        Monte Carlo pricing of European call with optional control variates
        
        Parameters:
        -----------
        control_variate: If True, uses Black-Scholes as control variate
        
        Returns:
        --------
        price: Option price
        If control_variate=True, returns (mc_price, cv_price)
        """
        S = self.simulate_gbm_vectorized(S0, T, r, sigma, N, M, antithetic)
        payoff = np.maximum(S[:, -1] - K, 0)
        mc_price = np.exp(-r*T) * np.mean(payoff)
        
        if not control_variate:
            return mc_price
        
        # Control Variate Implementation
        bs_price = self.black_scholes_call(S0, K, T, r, sigma)
        
        # Use a simpler European call as control (same parameters but fewer time steps)
        S_control = self.simulate_gbm_vectorized(S0, T, r, sigma, 1, M, antithetic)
        payoff_control = np.maximum(S_control[:, -1] - K, 0)
        mc_control = np.exp(-r*T) * payoff_control
        
        # Calculate optimal beta (correlation coefficient)
        beta = -np.cov(payoff, mc_control)[0, 1] / np.var(mc_control)
        
        # Control variate estimator
        cv_price = mc_price + beta * (np.exp(-r*T) * np.mean(mc_control) - bs_price)
        
        return mc_price, cv_price
    
    def european_put_mc(self, 
                       S0: float, 
                       K: float, 
                       T: float, 
                       r: float, 
                       sigma: float,
                       N: int = 100, 
                       M: int = 100000, 
                       antithetic: bool = True,
                       control_variate: bool = False) -> Union[float, Tuple[float, float]]:
        """Monte Carlo pricing of European put with optional control variates"""
        S = self.simulate_gbm_vectorized(S0, T, r, sigma, N, M, antithetic)
        payoff = np.maximum(K - S[:, -1], 0)
        mc_price = np.exp(-r*T) * np.mean(payoff)
        
        if not control_variate:
            return mc_price
        
        # Control Variate Implementation
        bs_price = self.black_scholes_put(S0, K, T, r, sigma)
        
        S_control = self.simulate_gbm_vectorized(S0, T, r, sigma, 1, M, antithetic)
        payoff_control = np.maximum(K - S_control[:, -1], 0)
        mc_control = np.exp(-r*T) * payoff_control
        
        beta = -np.cov(payoff, mc_control)[0, 1] / np.var(mc_control)
        cv_price = mc_price + beta * (np.exp(-r*T) * np.mean(mc_control) - bs_price)
        
        return mc_price, cv_price
    
    def convergence_analysis(self, 
                           option_func, 
                           true_price: float,
                           M_values: np.ndarray,
                           **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyze convergence of Monte Carlo estimator
        
        Parameters:
        -----------
        option_func: Pricing function to analyze
        true_price: Theoretical price for comparison
        M_values: Array of simulation counts to test
        **kwargs: Parameters for option_func
        
        Returns:
        --------
        prices: Array of MC prices for each M
        errors: Array of absolute errors
        """
        prices = np.zeros_like(M_values, dtype=float)
        errors = np.zeros_like(M_values, dtype=float)
        
        for i, M in enumerate(M_values):
            kwargs['M'] = int(M)
            price = option_func(**kwargs)
            
            # Handle control variate case
            if isinstance(price, tuple):
                price = price[1]  # Use control variate price
            
            prices[i] = price
            errors[i] = abs(price - true_price)
        
        return prices, errors


# Standalone functions for backward compatibility
def simulate_gbm(S0, T, r, sigma, N, M, antithetic=False, seed=None):
    """Backward compatible GBM simulation"""
    engine = MCPricingEngine(seed=seed)
    return engine.simulate_gbm_vectorized(S0, T, r, sigma, N, M, antithetic)

def black_scholes_call(S0, K, T, r, sigma):
    """Backward compatible Black-Scholes call"""
    engine = MCPricingEngine()
    return engine.black_scholes_call(S0, K, T, r, sigma)

def black_scholes_put(S0, K, T, r, sigma):
    """Backward compatible Black-Scholes put"""
    engine = MCPricingEngine()
    return engine.black_scholes_put(S0, K, T, r, sigma)

def european_call_mc(S0, K, T, r, sigma, N=100, M=100000, antithetic=True, seed=None):
    """Backward compatible European call MC"""
    engine = MCPricingEngine(seed=seed)
    return engine.european_call_mc(S0, K, T, r, sigma, N, M, antithetic)

def european_put_mc(S0, K, T, r, sigma, N=100, M=100000, antithetic=True, seed=None):
    """Backward compatible European put MC"""
    engine = MCPricingEngine(seed=seed)
    return engine.european_put_mc(S0, K, T, r, sigma, N, M, antithetic)


if __name__ == "__main__":
    # Performance benchmark
    import time
    
    # Parameters
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    M = 100000
    
    engine = MCPricingEngine(seed=42)
    
    print("=== INDUSTRIAL MONTE CARLO PRICING ENGINE ===")
    print(f"Parameters: S0={S0}, K={K}, T={T}, r={r}, σ={sigma}")
    print(f"Simulations: {M:,}")
    
    # Black-Scholes benchmark
    bs_call = engine.black_scholes_call(S0, K, T, r, sigma)
    bs_put = engine.black_scholes_put(S0, K, T, r, sigma)
    
    print(f"\n--- Theoretical Prices (Black-Scholes) ---")
    print(f"Call: {bs_call:.6f}")
    print(f"Put:  {bs_put:.6f}")
    
    # Monte Carlo without control variates
    start = time.time()
    mc_call = engine.european_call_mc(S0, K, T, r, sigma, M=M, antithetic=True)
    mc_time = time.time() - start
    
    print(f"\n--- Monte Carlo (Antithetic) ---")
    print(f"Call: {mc_call:.6f} (Error: {abs(mc_call - bs_call):.6f})")
    print(f"Time: {mc_time:.3f}s")
    
    # Monte Carlo with control variates
    start = time.time()
    mc_call_plain, mc_call_cv = engine.european_call_mc(S0, K, T, r, sigma, M=M, 
                                                       antithetic=True, control_variate=True)
    cv_time = time.time() - start
    
    print(f"\n--- Monte Carlo with Control Variates ---")
    print(f"Plain MC: {mc_call_plain:.6f} (Error: {abs(mc_call_plain - bs_call):.6f})")
    print(f"With CV:  {mc_call_cv:.6f} (Error: {abs(mc_call_cv - bs_call):.6f})")
    print(f"CV Improvement: {abs(mc_call_plain - bs_call) / abs(mc_call_cv - bs_call):.2f}x")
    print(f"Time: {cv_time:.3f}s")
    
    # Quick convergence test
    print(f"\n--- Convergence Analysis ---")
    M_values = np.array([1000, 5000, 10000, 50000, 100000])
    prices, errors = engine.convergence_analysis(
        engine.european_call_mc, bs_call, M_values,
        S0=S0, K=K, T=T, r=r, sigma=sigma, antithetic=True
    )
    
    for M, price, error in zip(M_values, prices, errors):
        print(f"M={M:6d}: Price={price:.6f}, Error={error:.6f}")
