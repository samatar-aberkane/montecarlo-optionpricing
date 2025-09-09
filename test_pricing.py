"""
Comprehensive Test Suite for Monte Carlo Option Pricing
Validation against analytical solutions and benchmarks
"""

import numpy as np
import unittest
import time
from typing import Dict, List, Tuple

from pricing import MCPricingEngine, black_scholes_call, black_scholes_put
from exotic_options import ExoticOptionsEngine
from greeks import GreeksEngine

class TestMCPricing(unittest.TestCase):
    """Test suite for Monte Carlo pricing accuracy and performance"""
    
    def setUp(self):
        """Set up test parameters"""
        self.engine = MCPricingEngine(seed=42)
        self.exotic_engine = ExoticOptionsEngine(seed=42)
        self.greeks_engine = GreeksEngine(seed=42)
        
        # Standard test parameters
        self.S0 = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.sigma = 0.2
        self.M = 100000  # High precision for tests
        
        # Tolerance levels
        self.tolerance_price = 0.5  # 50 cents for option prices
        self.tolerance_greeks = 0.05  # 5% relative error for Greeks
    
    def test_european_call_accuracy(self):
        """Test European call pricing accuracy vs Black-Scholes"""
        print("Testing European Call Accuracy...")
        
        bs_price = self.engine.black_scholes_call(self.S0, self.K, self.T, self.r, self.sigma)
        mc_price = self.engine.european_call_mc(self.S0, self.K, self.T, self.r, self.sigma, 
                                               M=self.M, antithetic=True)
        
        error = abs(mc_price - bs_price)
        print(f"  Black-Scholes: {bs_price:.6f}")
        print(f"  Monte Carlo:   {mc_price:.6f}")
        print(f"  Error:         {error:.6f}")
        
        self.assertLess(error, self.tolerance_price, 
                       f"European call error {error:.6f} exceeds tolerance {self.tolerance_price}")
    
    def test_european_put_accuracy(self):
        """Test European put pricing accuracy vs Black-Scholes"""
        print("Testing European Put Accuracy...")
        
        bs_price = self.engine.black_scholes_put(self.S0, self.K, self.T, self.r, self.sigma)
        mc_price = self.engine.european_put_mc(self.S0, self.K, self.T, self.r, self.sigma, 
                                              M=self.M, antithetic=True)
        
        error = abs(mc_price - bs_price)
        print(f"  Black-Scholes: {bs_price:.6f}")
        print(f"  Monte Carlo:   {mc_price:.6f}")
        print(f"  Error:         {error:.6f}")
        
        self.assertLess(error, self.tolerance_price)
    
    def test_control_variates_improvement(self):
        """Test that control variates reduce variance"""
        print("Testing Control Variates Effectiveness...")
        
        # Run multiple independent trials
        n_trials = 10
        mc_prices = []
        cv_prices = []
        
        for i in range(n_trials):
            engine = MCPricingEngine(seed=i)  # Different seed for each trial
            mc_price, cv_price = engine.european_call_mc(
                self.S0, self.K, self.T, self.r, self.sigma, 
                M=10000, antithetic=True, control_variate=True
            )
            mc_prices.append(mc_price)
            cv_prices.append(cv_price)
        
        # Calculate variances
        mc_var = np.var(mc_prices)
        cv_var = np.var(cv_prices)
        variance_reduction = mc_var / cv_var if cv_var > 0 else np.inf
        
        print(f"  Plain MC Variance:     {mc_var:.8f}")
        print(f"  Control Variate Var:   {cv_var:.8f}")
        print(f"  Variance Reduction:    {variance_reduction:.2f}x")
        
        self.assertGreater(variance_reduction, 1.5, 
                          "Control variates should reduce variance by at least 1.5x")
    
    def test_antithetic_variates(self):
        """Test antithetic variates variance reduction"""
        print("Testing Antithetic Variates...")
        
        n_trials = 20
        regular_prices = []
        antithetic_prices = []
        
        for i in range(n_trials):
            engine = MCPricingEngine(seed=i)
            
            regular = engine.european_call_mc(self.S0, self.K, self.T, self.r, self.sigma, 
                                            M=5000, antithetic=False)
            antithetic = engine.european_call_mc(self.S0, self.K, self.T, self.r, self.sigma, 
                                               M=5000, antithetic=True)
            
            regular_prices.append(regular)
            antithetic_prices.append(antithetic)
        
        regular_var = np.var(regular_prices)
        antithetic_var = np.var(antithetic_prices)
        
        print(f"  Regular Variance:      {regular_var:.8f}")
        print(f"  Antithetic Variance:   {antithetic_var:.8f}")
        print(f"  Improvement:           {regular_var / antithetic_var:.2f}x")
        
        self.assertLess(antithetic_var, regular_var, 
                       "Antithetic variates should reduce variance")
    
    def test_put_call_parity(self):
        """Test put-call parity relationship"""
        print("Testing Put-Call Parity...")
        
        call_price = self.engine.european_call_mc(self.S0, self.K, self.T, self.r, self.sigma, M=self.M)
        put_price = self.engine.european_put_mc(self.S0, self.K, self.T, self.r, self.sigma, M=self.M)
        
        # Put-call parity: C - P = S0 - K*exp(-rT)
        parity_lhs = call_price - put_price
        parity_rhs = self.S0 - self.K * np.exp(-self.r * self.T)
        parity_error = abs(parity_lhs - parity_rhs)
        
        print(f"  Call - Put:            {parity_lhs:.6f}")
        print(f"  S0 - K*exp(-rT):       {parity_rhs:.6f}")
        print(f"  Parity Error:          {parity_error:.6f}")
        
        self.assertLess(parity_error, 0.1, "Put-call parity violation too large")
    
    def test_greeks_accuracy(self):
        """Test Greeks accuracy vs analytical Black-Scholes"""
        print("Testing Greeks Accuracy...")
        
        from pricing import european_call_mc
        
        # Analytical Greeks
        analytical = self.greeks_engine.analytical_greeks_bs(
            self.S0, self.K, self.T, self.r, self.sigma, 'call'
        )
        
        # Monte Carlo Greeks
        params = {
            'S0': self.S0, 'K': self.K, 'T': self.T, 'r': self.r, 'sigma': self.sigma,
            'M': 50000, 'antithetic': True
        }
        
        mc_greeks = self.greeks_engine.compute_all_greeks(european_call_mc, params, 'finite_diff')
        
        for greek_name in ['delta', 'gamma', 'vega', 'theta', 'rho']:
            analytical_val = analytical[greek_name]
            mc_val = mc_greeks[greek_name]
            
            if abs(analytical_val) > 1e-6:  # Avoid division by zero
                relative_error = abs(mc_val - analytical_val) / abs(analytical_val)
            else:
                relative_error = abs(mc_val - analytical_val)
            
            print(f"  {greek_name.capitalize():5s}: Analytical={analytical_val:8.6f}, MC={mc_val:8.6f}, Error={relative_error:.4f}")
            
            tolerance = self.tolerance_greeks * 2 if greek_name in ['gamma', 'theta'] else self.tolerance_greeks
            self.assertLess(relative_error, tolerance, 
                           f"{greek_name} relative error {relative_error:.4f} exceeds tolerance {tolerance}")
    
    def test_asian_option_bounds(self):
        """Test Asian option bounds and monotonicity"""
        print("Testing Asian Option Properties...")
        
        # Asian call should be less than European call (Jensen's inequality)
        european_call = self.engine.european_call_mc(self.S0, self.K, self.T, self.r, self.sigma, M=50000)
        asian_call = self.exotic_engine.asian_call_mc(self.S0, self.K, self.T, self.r, self.sigma, M=50000)
        
        print(f"  European Call:         {european_call:.6f}")
        print(f"  Asian Call:            {asian_call:.6f}")
        
        self.assertLess(asian_call, european_call + 0.5,  # Small tolerance for MC error
                       "Asian call should be less than European call")
        self.assertGreater(asian_call, 0, "Asian call should be positive")
    
    def test_barrier_option_bounds(self):
        """Test barrier option properties"""
        print("Testing Barrier Option Properties...")
        
        B = 120  # Barrier above current price
        european_call = self.engine.european_call_mc(self.S0, self.K, self.T, self.r, self.sigma, M=50000)
        barrier_call = self.exotic_engine.barrier_up_out_call_mc(
            self.S0, self.K, B, self.T, self.r, self.sigma, M=50000
        )
        
        print(f"  European Call:         {european_call:.6f}")
        print(f"  Up-and-Out Call (B={B}): {barrier_call:.6f}")
        
        self.assertLess(barrier_call, european_call,
                       "Barrier call should be less than European call")
        self.assertGreaterEqual(barrier_call, 0, "Barrier call should be non-negative")
    
    def test_convergence_rates(self):
        """Test Monte Carlo convergence rates"""
        print("Testing Convergence Rates...")
        
        M_values = np.array([1000, 5000, 10000, 50000, 100000])
        true_price = self.engine.black_scholes_call(self.S0, self.K, self.T, self.r, self.sigma)
        
        prices, errors = self.engine.convergence_analysis(
            self.engine.european_call_mc, true_price, M_values,
            S0=self.S0, K=self.K, T=self.T, r=self.r, sigma=self.sigma, antithetic=True
        )
        
        print(f"  True Price: {true_price:.6f}")
        for M, price, error in zip(M_values, prices, errors):
            print(f"  M={M:6d}: Price={price:.6f}, Error={error:.6f}")
        
        # Test that errors generally decrease (allowing for some MC noise)
        # Use a simple trend test
        log_M = np.log(M_values)
        log_errors = np.log(np.maximum(errors, 1e-8))  # Avoid log(0)
        
        # Fit line: should have negative slope (errors decrease with M)
        slope = np.polyfit(log_M, log_errors, 1)[0]
        print(f"  Convergence Slope:     {slope:.4f} (should be ≈ -0.5)")
        
        self.assertLess(slope, -0.2, "Monte Carlo should show convergence")
    
    def test_performance_benchmarks(self):
        """Test performance requirements"""
        print("Testing Performance Benchmarks...")
        
        # European option should price quickly
        start_time = time.time()
        price = self.engine.european_call_mc(self.S0, self.K, self.T, self.r, self.sigma, M=100000)
        european_time = time.time() - start_time
        
        print(f"  European Call (100K):  {european_time:.3f}s")
        
        # Asian option benchmark
        start_time = time.time()
        asian_price = self.exotic_engine.asian_call_mc(self.S0, self.K, self.T, self.r, self.sigma, M=50000)
        asian_time = time.time() - start_time
        
        print(f"  Asian Call (50K):      {asian_time:.3f}s")
        
        # Performance requirements (adjust based on your hardware)
        self.assertLess(european_time, 5.0, "European option pricing too slow")
        self.assertLess(asian_time, 10.0, "Asian option pricing too slow")


class TestRegressionBenchmarks(unittest.TestCase):
    """Regression tests against known benchmark values"""
    
    def setUp(self):
        self.engine = MCPricingEngine(seed=42)
        self.exotic_engine = ExoticOptionsEngine(seed=42)
    
    def test_benchmark_european_options(self):
        """Test against published benchmark values"""
        print("Testing Against Published Benchmarks...")
        
        # Standard benchmark from Haug "Option Pricing Formulas"
        test_cases = [
            # S0, K, T, r, sigma, expected_call, expected_put
            (100, 100, 1.0, 0.05, 0.20, 10.4506, 5.5735),
            (100, 110, 1.0, 0.05, 0.20, 4.2294, 9.3523),
            (100, 90, 1.0, 0.05, 0.20, 18.8630, 4.0399),
        ]
        
        for S0, K, T, r, sigma, expected_call, expected_put in test_cases:
            call_price = self.engine.european_call_mc(S0, K, T, r, sigma, M=100000, antithetic=True)
            put_price = self.engine.european_put_mc(S0, K, T, r, sigma, M=100000, antithetic=True)
            
            call_error = abs(call_price - expected_call)
            put_error = abs(put_price - expected_put)
            
            print(f"  S0={S0}, K={K}: Call={call_price:.4f} (exp={expected_call:.4f}, err={call_error:.4f})")
            print(f"  S0={S0}, K={K}: Put={put_price:.4f} (exp={expected_put:.4f}, err={put_error:.4f})")
            
            self.assertLess(call_error, 0.2, f"Call benchmark failed for S0={S0}, K={K}")
            self.assertLess(put_error, 0.2, f"Put benchmark failed for S0={S0}, K={K}")


def run_performance_suite():
    """Run comprehensive performance analysis"""
    print("=" * 60)
    print("
