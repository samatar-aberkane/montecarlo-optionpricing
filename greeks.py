"""
Greeks Calculation
Advanced finite differences with optimal step sizes and pathwise Greeks
"""

import numpy as np
from typing import Dict, Optional, Callable, Any
from pricing import MCPricingEngine, black_scholes_call, black_scholes_put
from scipy.stats import norm

class GreeksEngine(MCPricingEngine):
    """
    Advanced Greeks calculation with multiple methodologies
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        # Optimal step sizes (from numerical analysis literature)
        self.optimal_eps = {
            'delta': 1e-4,
            'gamma': 1e-4,
            'vega': 1e-4,
            'theta': 1e-4,  # In years
            'rho': 1e-4
        }
    
    def finite_difference_greeks(self, 
                                pricing_func: Callable,
                                params: Dict[str, Any],
                                greek_type: str,
                                method: str = 'central') -> float:
        """
        Compute Greeks using finite differences with optimal step sizes
        
        Parameters:
        -----------
        pricing_func: Pricing function to differentiate
        params: Parameters dictionary
        greek_type: 'delta', 'gamma', 'vega', 'theta', 'rho'
        method: 'forward', 'backward', 'central'
        """
        eps = self.optimal_eps[greek_type]
        base_price = pricing_func(**params)
        
        # Handle tuple return (MC with control variates)
        if isinstance(base_price, tuple):
            base_price = base_price[0]  # Use plain MC for stability
        
        if greek_type == 'delta':
            param_name = 'S0'
        elif greek_type == 'gamma':
            return self._gamma_finite_diff(pricing_func, params, eps)
        elif greek_type == 'vega':
            param_name = 'sigma'
            eps = 0.01  # 1% volatility shift
        elif greek_type == 'theta':
            param_name = 'T'
            eps = 1/365  # 1 day
        elif greek_type == 'rho':
            param_name = 'r'
            eps = 0.01  # 1% rate shift
        else:
            raise ValueError(f"Unknown greek type: {greek_type}")
        
        if method == 'central':
            # Central difference (most accurate)
            params_up = params.copy()
            params_down = params.copy()
            params_up[param_name] += eps
            params_down[param_name] -= eps
            
            price_up = pricing_func(**params_up)
            price_down = pricing_func(**params_down)
            
            if isinstance(price_up, tuple):
                price_up = price_up[0]
            
            greek = (price_up - base_price) / eps
            
        elif method == 'backward':
            params_down = params.copy()
            params_down[param_name] -= eps
            price_down = pricing_func(**params_down)
            
            if isinstance(price_down, tuple):
                price_down = price_down[0]
            
            greek = (base_price - price_down) / eps
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Handle special cases
        if greek_type == 'theta':
            greek = -greek  # Theta is negative (time decay)
        
        return greek
    
    def _gamma_finite_diff(self, pricing_func: Callable, params: Dict[str, Any], eps: float) -> float:
        """Gamma requires second derivative (central difference)"""
        params_up = params.copy()
        params_down = params.copy()
        params_up['S0'] += eps
        params_down['S0'] -= eps
        
        price_up = pricing_func(**params_up)
        price = pricing_func(**params)
        price_down = pricing_func(**params_down)
        
        # Handle tuples
        if isinstance(price_up, tuple):
            price_up = price_up[0]
        if isinstance(price, tuple):
            price = price[0]
        if isinstance(price_down, tuple):
            price_down = price_down[0]
        
        return (price_up - 2*price + price_down) / (eps**2)
    
    def pathwise_delta(self, 
                      S0: float, 
                      K: float, 
                      T: float, 
                      r: float, 
                      sigma: float,
                      option_type: str = 'call',
                      N: int = 100, 
                      M: int = 100000, 
                      antithetic: bool = True) -> float:
        """
        Pathwise delta (more accurate for smooth payoffs)
        """
        S = self.simulate_gbm_vectorized(S0, T, r, sigma, N, M, antithetic)
        
        if option_type.lower() == 'call':
            # Delta = exp(-rT) * E[1_{S_T > K} * S_T / S0]
            in_money = S[:, -1] > K
            pathwise_deriv = np.where(in_money, S[:, -1] / S0, 0)
        elif option_type.lower() == 'put':
            # Delta = exp(-rT) * E[-1_{S_T < K} * S_T / S0]
            in_money = S[:, -1] < K
            pathwise_deriv = np.where(in_money, -S[:, -1] / S0, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        return np.exp(-r*T) * np.mean(pathwise_deriv)
    
    def likelihood_ratio_vega(self, 
                             S0: float, 
                             K: float, 
                             T: float, 
                             r: float, 
                             sigma: float,
                             option_type: str = 'call',
                             N: int = 100, 
                             M: int = 100000, 
                             antithetic: bool = True) -> float:
        """
        Likelihood ratio method for Vega (more stable)
        """
        S = self.simulate_gbm_vectorized(S0, T, r, sigma, N, M, antithetic)
        
        if option_type.lower() == 'call':
            payoff = np.maximum(S[:, -1] - K, 0)
        elif option_type.lower() == 'put':
            payoff = np.maximum(K - S[:, -1], 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        # Likelihood ratio weight for volatility
        W_T = np.sum(self.rng.normal(0, 1, (M, N)), axis=1)  # Sum of Brownian increments
        
        # Weight for vega: d/dσ log(likelihood)
        weight = (np.log(S[:, -1] / S0) - (r - 0.5*sigma**2)*T) / sigma - sigma*T
        
        return np.exp(-r*T) * np.mean(payoff * weight)
    
    def analytical_greeks_bs(self, 
                           S0: float, 
                           K: float, 
                           T: float, 
                           r: float, 
                           sigma: float,
                           option_type: str = 'call') -> Dict[str, float]:
        """
        Analytical Black-Scholes Greeks for comparison
        """
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}
            
        d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        phi_d1 = norm.pdf(d1)
        phi_d2 = norm.pdf(d2)
        Phi_d1 = norm.cdf(d1)
        Phi_d2 = norm.cdf(d2)
        
        if option_type.lower() == 'call':
            delta = Phi_d1
            theta = (-S0*phi_d1*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*Phi_d2) / 365
            rho = K*T*np.exp(-r*T)*Phi_d2 / 100
        else:  # put
            delta = Phi_d1 - 1
            theta = (-S0*phi_d1*sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*(1-Phi_d2)) / 365
            rho = -K*T*np.exp(-r*T)*(1-Phi_d2) / 100
        
        gamma = phi_d1 / (S0*sigma*np.sqrt(T))
        vega = S0*phi_d1*np.sqrt(T) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }
    
    def compute_all_greeks(self, 
                          pricing_func: Callable,
                          params: Dict[str, Any],
                          method: str = 'finite_diff') -> Dict[str, float]:
        """
        Compute all Greeks using specified method
        
        Parameters:
        -----------
        method: 'finite_diff', 'pathwise', 'likelihood_ratio', 'analytical'
        """
        greeks = {}
        
        if method == 'finite_diff':
            for greek in ['delta', 'gamma', 'vega', 'theta', 'rho']:
                greeks[greek] = self.finite_difference_greeks(pricing_func, params, greek)
                
        elif method == 'analytical' and 'option_type' in params:
            # Only for vanilla options
            greeks = self.analytical_greeks_bs(
                params['S0'], params['K'], params['T'], 
                params['r'], params['sigma'], params['option_type']
            )
            
        elif method == 'pathwise':
            # Implement pathwise for other Greeks as needed
            greeks['delta'] = self.pathwise_delta(
                params['S0'], params['K'], params['T'], params['r'], params['sigma'],
                params.get('option_type', 'call'), params.get('N', 100), 
                params.get('M', 100000)
            )
            # Fill others with finite differences
            for greek in ['gamma', 'vega', 'theta', 'rho']:
                greeks[greek] = self.finite_difference_greeks(pricing_func, params, greek)
                
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return greeks


# Backward compatibility functions
def delta_mc(option_type, S0, K, T, r, sigma, eps=1e-4, **kwargs):
    """Backward compatible delta calculation"""
    from pricing import european_call_mc, european_put_mc
    engine = GreeksEngine(kwargs.get('seed'))
    
    pricing_func = european_call_mc if option_type.lower() == 'call' else european_put_mc
    params = {'S0': S0, 'K': K, 'T': T, 'r': r, 'sigma': sigma, **kwargs}
    
    return engine.finite_difference_greeks(pricing_func, params, 'delta')

def gamma_mc(option_type, S0, K, T, r, sigma, eps=1e-4, **kwargs):
    """Backward compatible gamma calculation"""
    from pricing import european_call_mc, european_put_mc
    engine = GreeksEngine(kwargs.get('seed'))
    
    pricing_func = european_call_mc if option_type.lower() == 'call' else european_put_mc
    params = {'S0': S0, 'K': K, 'T': T, 'r': r, 'sigma': sigma, **kwargs}
    
    return engine.finite_difference_greeks(pricing_func, params, 'gamma')

def vega_mc(option_type, S0, K, T, r, sigma, eps=1e-4, **kwargs):
    """Backward compatible vega calculation"""
    from pricing import european_call_mc, european_put_mc
    engine = GreeksEngine(kwargs.get('seed'))
    
    pricing_func = european_call_mc if option_type.lower() == 'call' else european_put_mc
    params = {'S0': S0, 'K': K, 'T': T, 'r': r, 'sigma': sigma, **kwargs}
    
    return engine.finite_difference_greeks(pricing_func, params, 'vega')

def theta_mc(option_type, S0, K, T, r, sigma, eps=1e-4, **kwargs):
    """Backward compatible theta calculation"""
    from pricing import european_call_mc, european_put_mc
    engine = GreeksEngine(kwargs.get('seed'))
    
    pricing_func = european_call_mc if option_type.lower() == 'call' else european_put_mc
    params = {'S0': S0, 'K': K, 'T': T, 'r': r, 'sigma': sigma, **kwargs}
    
    return engine.finite_difference_greeks(pricing_func, params, 'theta')

def rho_mc(option_type, S0, K, T, r, sigma, eps=1e-4, **kwargs):
    """Backward compatible rho calculation"""
    from pricing import european_call_mc, european_put_mc
    engine = GreeksEngine(kwargs.get('seed'))
    
    pricing_func = european_call_mc if option_type.lower() == 'call' else european_put_mc
    params = {'S0': S0, 'K': K, 'T': T, 'r': r, 'sigma': sigma, **kwargs}
    
    return engine.finite_difference_greeks(pricing_func, params, 'rho')


if __name__ == "__main__":
    import time
    from pricing import european_call_mc, european_put_mc

    print("=== GREEKS CALCULATION ENGINE ===")

    # Test parameters
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    M = 50000

    engine = GreeksEngine(seed=42)

    print(f"Parameters: S0={S0}, K={K}, T={T}, r={r}, σ={sigma}")
    print(f"Simulations: {M:,}")

    # Analytical Greeks (benchmark)
    analytical = engine.analytical_greeks_bs(S0, K, T, r, sigma, 'call')
    print(f"\n--- Analytical Greeks (Black-Scholes) ---")
    for greek, value in analytical.items():
        print(f"{greek.capitalize()}: {value:.6f}")

    # Monte Carlo Greeks
    params = {
        'S0': S0, 'K': K, 'T': T, 'r': r, 'sigma': sigma,
        'M': M, 'antithetic': True, 'option_type': 'call'
    }

    print(f"\n--- Monte Carlo Greeks (Finite Differences) ---")
    start = time.time()
    mc_greeks = engine.compute_all_greeks(european_call_mc, params, 'finite_diff')
    fd_time = time.time() - start

    for greek, value in mc_greeks.items():
        analytical_val = analytical[greek]
        error = abs(value - analytical_val)
        print(f"{greek.capitalize()}: {value:.6f} (Error: {error:.6f})")

    print(f"Time: {fd_time:.3f}s")

    # Pathwise Greeks
    print(f"\n--- Advanced Greeks (Pathwise Delta) ---")
    start = time.time()
    pathwise_delta = engine.pathwise_delta(S0, K, T, r, sigma, 'call', M=M)
    pathwise_time = time.time() - start

    delta_error = abs(pathwise_delta - analytical['delta'])
    print(f"Pathwise Delta: {pathwise_delta:.6f} (Error: {delta_error:.6f})")
    print(f"Time: {pathwise_time:.3f}s")

    # Likelihood ratio vega
    start = time.time()
    lr_vega = engine.likelihood_ratio_vega(S0, K, T, r, sigma, 'call', M=M)
    lr_time = time.time() - start

    vega_error = abs(lr_vega - analytical['vega'])
    print(f"Likelihood Ratio Vega: {lr_vega:.6f} (Error: {vega_error:.6f})")
    print(f"Time: {lr_time:.3f}s")

    # Performance summary
    print(f"\n--- Performance Comparison ---")
    print(f"Finite Diff:   {fd_time:.3f}s (all Greeks)")
    print(f"Pathwise:      {pathwise_time:.3f}s (delta only)")
    print(f"Likelihood:    {lr_time:.3f}s (vega only)")
            if isinstance(price_down, tuple):
                price_down = price_down[0]
            
            greek = (price_up - price_down) / (2 * eps)
            
        elif method == 'forward':
            params_up = params.copy()
            params_up[param_name] += eps
            price_up = pricing_func(**params_up)
            
            if isinstance(price_up, tuple):
                price_
