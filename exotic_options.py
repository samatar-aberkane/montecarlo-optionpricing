"""
Exotic Options Pricing
Advanced Monte Carlo with Control Variates for Path-Dependent Options
"""

import numpy as np
from typing import Optional, Tuple, Union
from pricing import MCPricingEngine, black_scholes_call, black_scholes_put

class ExoticOptionsEngine(MCPricingEngine):
    """
    Exotic options pricing with advanced variance reduction techniques
    """
    
    def asian_call_mc(self, 
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
        Asian call option pricing with control variates
        
        Uses geometric average Asian option as control variate
        """
        S = self.simulate_gbm_vectorized(S0, T, r, sigma, N, M, antithetic)
        
        # Arithmetic average (target)
        avg_arith = np.mean(S[:, 1:], axis=1)
        payoff_arith = np.maximum(avg_arith - K, 0)
        mc_price = np.exp(-r*T) * np.mean(payoff_arith)
        
        if not control_variate:
            return mc_price
        
        # Geometric average (control variate)
        # More stable numerically using log-space
        log_avg_geom = np.mean(np.log(S[:, 1:]), axis=1)
        avg_geom = np.exp(log_avg_geom)
        payoff_geom = np.maximum(avg_geom - K, 0)
        
        # Analytical price for geometric Asian (Kemna-Vorst formula)
        sigma_g = sigma * np.sqrt((N + 1) * (2*N + 1) / (6 * N**2))
        rho_g = 0.5 * (r - 0.5 * sigma**2 + sigma_g**2)
        
        # Adjusted parameters for geometric Asian
        S0_adj = S0 * np.exp((rho_g - r) * T)
        geom_price = np.exp(-rho_g * T) * black_scholes_call(S0_adj, K, T, rho_g, sigma_g)
        
        # Control variate adjustment
        beta = -np.cov(payoff_arith, payoff_geom)[0, 1] / np.var(payoff_geom)
        cv_price = mc_price + beta * (np.exp(-r*T) * np.mean(payoff_geom) - geom_price)
        
        return mc_price, cv_price
    
    def asian_put_mc(self, 
                    S0: float, 
                    K: float, 
                    T: float, 
                    r: float, 
                    sigma: float,
                    N: int = 100, 
                    M: int = 100000, 
                    antithetic: bool = True,
                    control_variate: bool = False) -> Union[float, Tuple[float, float]]:
        """Asian put option pricing with geometric control variate"""
        S = self.simulate_gbm_vectorized(S0, T, r, sigma, N, M, antithetic)
        
        # Arithmetic average
        avg_arith = np.mean(S[:, 1:], axis=1)
        payoff_arith = np.maximum(K - avg_arith, 0)
        mc_price = np.exp(-r*T) * np.mean(payoff_arith)
        
        if not control_variate:
            return mc_price
        
        # Geometric average control
        log_avg_geom = np.mean(np.log(S[:, 1:]), axis=1)
        avg_geom = np.exp(log_avg_geom)
        payoff_geom = np.maximum(K - avg_geom, 0)
        
        # Analytical geometric Asian put
        sigma_g = sigma * np.sqrt((N + 1) * (2*N + 1) / (6 * N**2))
        rho_g = 0.5 * (r - 0.5 * sigma**2 + sigma_g**2)
        S0_adj = S0 * np.exp((rho_g - r) * T)
        geom_price = np.exp(-rho_g * T) * black_scholes_put(S0_adj, K, T, rho_g, sigma_g)
        
        beta = -np.cov(payoff_arith, payoff_geom)[0, 1] / np.var(payoff_geom)
        cv_price = mc_price + beta * (np.exp(-r*T) * np.mean(payoff_geom) - geom_price)
        
        return mc_price, cv_price
    
    def barrier_up_out_call_mc(self, 
                              S0: float, 
                              K: float, 
                              B: float, 
                              T: float, 
                              r: float, 
                              sigma: float,
                              N: int = 100, 
                              M: int = 100000, 
                              antithetic: bool = True,
                              control_variate: bool = False) -> Union[float, Tuple[float, float]]:
        """
        Up-and-Out barrier call with control variates
        """
        S = self.simulate_gbm_vectorized(S0, T, r, sigma, N, M, antithetic)
        
        # Check barrier breach
        breached = np.any(S[:, 1:] >= B, axis=1)
        payoff = np.where(breached, 0, np.maximum(S[:, -1] - K, 0))
        mc_price = np.exp(-r*T) * np.mean(payoff)
        
        if not control_variate:
            return mc_price
        
        # Use European call as control variate
        payoff_european = np.maximum(S[:, -1] - K, 0)
        european_price = black_scholes_call(S0, K, T, r, sigma)
        
        # Improve control by using barrier probability
        prob_no_breach = np.mean(~breached)
        
        beta = -np.cov(payoff, payoff_european)[0, 1] / np.var(payoff_european)
        cv_price = mc_price + beta * (np.exp(-r*T) * np.mean(payoff_european) - european_price)
        
        return mc_price, cv_price
    
    def barrier_down_out_put_mc(self, 
                               S0: float, 
                               K: float, 
                               B: float, 
                               T: float, 
                               r: float, 
                               sigma: float,
                               N: int = 100, 
                               M: int = 100000, 
                               antithetic: bool = True,
                               control_variate: bool = False) -> Union[float, Tuple[float, float]]:
        """Down-and-Out barrier put with control variates"""
        S = self.simulate_gbm_vectorized(S0, T, r, sigma, N, M, antithetic)
        
        # Check barrier breach
        breached = np.any(S[:, 1:] <= B, axis=1)
        payoff = np.where(breached, 0, np.maximum(K - S[:, -1], 0))
        mc_price = np.exp(-r*T) * np.mean(payoff)
        
        if not control_variate:
            return mc_price
        
        # European put control variate
        payoff_european = np.maximum(K - S[:, -1], 0)
        european_price = black_scholes_put(S0, K, T, r, sigma)
        
        beta = -np.cov(payoff, payoff_european)[0, 1] / np.var(payoff_european)
        cv_price = mc_price + beta * (np.exp(-r*T) * np.mean(payoff_european) - european_price)
        
        return mc_price, cv_price
    
    def lookback_call_mc(self, 
                        S0: float, 
                        T: float, 
                        r: float, 
                        sigma: float,
                        N: int = 100, 
                        M: int = 100000, 
                        antithetic: bool = True,
                        control_variate: bool = False) -> Union[float, Tuple[float, float]]:
        """
        Floating strike lookback call: S_T - min(S_t)
        """
        S = self.simulate_gbm_vectorized(S0, T, r, sigma, N, M, antithetic)
        
        # Lookback payoff
        S_min = np.min(S[:, 1:], axis=1)
        payoff = S[:, -1] - S_min
        mc_price = np.exp(-r*T) * np.mean(payoff)
        
        if not control_variate:
            return mc_price
        
        # Use S_T as control (perfect correlation but different variance)
        payoff_control = S[:, -1]
        
        # Analytical expectation: E[S_T] = S0 * exp(rT)
        expected_ST = S0 * np.exp(r * T)
        
        beta = -np.cov(payoff, payoff_control)[0, 1] / np.var(payoff_control)
        cv_price = mc_price + beta * (np.exp(-r*T) * np.mean(payoff_control) - np.exp(-r*T) * expected_ST)
        
        return mc_price, cv_price


# Backward compatibility functions
def asian_call_mc(S0, K, T, r, sigma, N=100, M=100000, antithetic=True, seed=None):
    """Backward compatible Asian call"""
    engine = ExoticOptionsEngine(seed=seed)
    return engine.asian_call_mc(S0, K, T, r, sigma, N, M, antithetic)

def asian_put_mc(S0, K, T, r, sigma, N=100, M=100000, antithetic=True, seed=None):
    """Backward compatible Asian put"""
    engine = ExoticOptionsEngine(seed=seed)
    return engine.asian_put_mc(S0, K, T, r, sigma, N, M, antithetic)

def barrier_call_mc(S0, K, B, T, r, sigma, N=100, M=100000, antithetic=True, seed=None):
    """Backward compatible barrier call (up-and-out)"""
    engine = ExoticOptionsEngine(seed=seed)
    return engine.barrier_up_out_call_mc(S0, K, B, T, r, sigma, N, M, antithetic)


if __name__ == "__main__":
    import time
    
    print("=== EXOTIC OPTIONS PRICING ENGINE ===")
    
    # Test parameters
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    B = 120  # Barrier level
    M = 100000
    
    engine = ExoticOptionsEngine(seed=42)
    
    print(f"Parameters: S0={S0}, K={K}, T={T}, r={r}, σ={sigma}")
    print(f"Barrier: {B}, Simulations: {M:,}")
    
    # Asian Options
    print(f"\n--- Asian Call Option ---")
    start = time.time()
    asian_plain, asian_cv = engine.asian_call_mc(S0, K, T, r, sigma, M=M, 
                                                antithetic=True, control_variate=True)
    asian_time = time.time() - start
    
    print(f"Plain MC:  {asian_plain:.6f}")
    print(f"With CV:   {asian_cv:.6f}")
    print(f"Time:      {asian_time:.3f}s")
    
    # Barrier Options
    print(f"\n--- Up-and-Out Barrier Call ---")
    start = time.time()
    barrier_plain, barrier_cv = engine.barrier_up_out_call_mc(S0, K, B, T, r, sigma, M=M,
                                                             antithetic=True, control_variate=True)
    barrier_time = time.time() - start
    
    print(f"Plain MC:  {barrier_plain:.6f}")
    print(f"With CV:   {barrier_cv:.6f}")
    print(f"Time:      {barrier_time:.3f}s")
    
    # Lookback Options
    print(f"\n--- Lookback Call ---")
    start = time.time()
    lookback_plain, lookback_cv = engine.lookback_call_mc(S0, T, r, sigma, M=M,
                                                         antithetic=True, control_variate=True)
    lookback_time = time.time() - start
    
    print(f"Plain MC:  {lookback_plain:.6f}")
    print(f"With CV:   {lookback_cv:.6f}")
    print(f"Time:      {lookback_time:.3f}s")
    
    # Performance comparison
    print(f"\n--- Performance Summary ---")
    print(f"Asian:     {asian_time:.3f}s")
    print(f"Barrier:   {barrier_time:.3f}s")
    print(f"Lookback:  {lookback_time:.3f}s")
