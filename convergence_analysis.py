"""
Convergence Analysis and Visualization for Monte Carlo Option Pricing
Analysis with statistical metrics and publication-ready plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Callable
import time
import pandas as pd

from pricing import MCPricingEngine, black_scholes_call, black_scholes_put
from exotic_options import ExoticOptionsEngine
from greeks import GreeksEngine

# Set style for publication-ready plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class ConvergenceAnalyzer:
    """
    Advanced convergence analysis for Monte Carlo methods
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.engine = MCPricingEngine(seed=seed)
        self.exotic_engine = ExoticOptionsEngine(seed=seed)
        self.greeks_engine = GreeksEngine(seed=seed)
    
   def monte_carlo_convergence(self, 
                               pricing_func: Callable,
                               true_price: float,
                               M_values: np.ndarray,
                               n_trials: int = 10,
                               **kwargs) -> Dict[str, np.ndarray]:
        """
        Analyze Monte Carlo convergence with statistical confidence
        
        Parameters:
        -----------
        pricing_func: Function to analyze
        true_price: Theoretical benchmark price
        M_values: Array of simulation counts
        n_trials: Number of independent trials for each M value
        
        Returns:
        --------
        Dictionary with arrays for mean prices, RMSE, std dev, and computation times
        """
        
        mean_prices = np.zeros(len(M_values))
        std_devs = np.zeros(len(M_values))
        rmses = np.zeros(len(M_values))
        computation_times = np.zeros(len(M_values))
        
        for i, M in enumerate(M_values):
            prices_per_trial = np.zeros(n_trials)
            start_time = time.time()
            
            kwargs_trial = kwargs.copy()
            kwargs_trial['M'] = int(M)
            
            for j in range(n_trials):
                # Call the pricing function
                price_result = pricing_func(**kwargs_trial)
                
                # Handle tuple return (e.g., from control variates)
                if isinstance(price_result, tuple):
                    prices_per_trial[j] = price_result[0]
                else:
                    prices_per_trial[j] = price_result
            
            end_time = time.time()
            
            mean_prices[i] = np.mean(prices_per_trial)
            std_devs[i] = np.std(prices_per_trial, ddof=1)
            rmses[i] = np.sqrt(np.mean((prices_per_trial - true_price)**2))
            computation_times[i] = (end_time - start_time) / n_trials
            
        return {
            'mean_prices': mean_prices,
            'std_devs': std_devs,
            'rmse': rmses,
            'computation_times': computation_times
        }
    
    def control_variates_analysis(self, 
                                 pricing_func: Callable,
                                 true_price: float,
                                 M: int = 50000,
                                 n_trials: int = 50,
                                 **kwargs) -> Dict[str, float]:
        """
        Analyze control variates effectiveness
        """
        plain_prices = []
        cv_prices = []
        plain_times = []
        cv_times = []
        
        for trial in range(n_trials):
            engine = MCPricingEngine(seed=trial * 42)
            
            # Plain Monte Carlo
            start_time = time.time()
            kwargs_plain = kwargs.copy()
            kwargs_plain['M'] = M
            kwargs_plain['control_variate'] = False
            
            price_plain = pricing_func(**kwargs_plain)
            if isinstance(price_plain, tuple):
                price_plain = price_plain[0]
            
            plain_time = time.time() - start_time
            plain_prices.append(price_plain)
            plain_times.append(plain_time)
            
            # Control variates
            start_time = time.time()
            kwargs_cv = kwargs.copy()
            kwargs_cv['M'] = M
            kwargs_cv['control_variate'] = True
            
            result_cv = pricing_func(**kwargs_cv)
            if isinstance(result_cv, tuple):
                price_cv = result_cv[1]  # Control variate price
            else:
                price_cv = result_cv
            
            cv_time = time.time() - start_time
            cv_prices.append(price_cv)
            cv_times.append(cv_time)
        
        # Calculate metrics
        plain_var = np.var(plain_prices)
        cv_var = np.var(cv_prices)
        variance_reduction = plain_var / cv_var if cv_var > 0 else np.inf
        
        plain_mse = np.mean([(p - true_price)**2 for p in plain_prices])
        cv_mse = np.mean([(p - true_price)**2 for p in cv_prices])
        mse_reduction = plain_mse / cv_mse if cv_mse > 0 else np.inf
        
        efficiency = variance_reduction / (np.mean(cv_times) / np.mean(plain_times))
        
        return {
            'plain_variance': plain_var,
            'cv_variance': cv_var,
            'variance_reduction': variance_reduction,
            'plain_mse': plain_mse,
            'cv_mse': cv_mse,
            'mse_reduction': mse_reduction,
            'plain_time': np.mean(plain_times),
            'cv_time': np.mean(cv_times),
            'time_overhead': np.mean(cv_times) / np.mean(plain_times),
            'efficiency': efficiency
        }
    
    def plot_convergence(self, 
                        results: Dict[str, np.ndarray],
                        true_price: float,
                        title: str = "Monte Carlo Convergence",
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Create publication-ready convergence plots
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        M_values = results['M_values']
        
        # 1. Price convergence
        ax1.errorbar(M_values, results['mean_prices'], 
                    yerr=results['std_prices'], 
                    marker='o', capsize=5, capthick=2, label='MC Estimates')
        ax1.axhline(y=true_price, color='red', linestyle='--', 
                   linewidth=2, label=f'True Price = {true_price:.6f}')
        ax1.set_xscale('log')
        ax1.set_xlabel('Number of Simulations (M)')
        ax1.set_ylabel('Option Price')
        ax1.set_title('Price Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Error convergence (log-log scale)
        ax2.loglog(M_values, results['rmse'], 'o-', linewidth=2, markersize=8, label='RMSE')
        ax2.loglog(M_values, results['mean_errors'], 's-', linewidth=2, markersize=6, label='Mean |Error|')
        
        # Theoretical 1/√M convergence
        theoretical_rate = results['rmse'][0] * np.sqrt(M_values[0] / M_values)
        ax2.loglog(M_values, theoretical_rate, 'k--', alpha=0.7, label='Theoretical O(1/√M)')
        
        ax2.set_xlabel('Number of Simulations (M)')
        ax2.set_ylabel('Error')
        ax2.set_title('Error Convergence (Log-Log Scale)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Variance reduction
        variance = results['std_prices']**2
        ax3.loglog(M_values, variance, 'o-', linewidth=2, markersize=8, label='Empirical Variance')
        theoretical_var = variance[0] * M_values[0] / M_values
        ax3.loglog(M_values, theoretical_var, 'k--', alpha=0.7, label='Theoretical O(1/M)')
        ax3.set_xlabel('Number of Simulations (M)')
        ax3.set_ylabel('Variance')
        ax3.set_title('Variance Reduction')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Computational efficiency
        efficiency = results['mean_errors'][0] / results['mean_errors']  # Improvement factor
        computational_cost = M_values * results['computation_times'] / results['computation_times'][0]
        
        ax4.loglog(computational_cost, results['mean_errors'], 'o-', linewidth=2, markersize=8)
        ax4.set_xlabel('Relative Computational Cost')
        ax4.set_ylabel('Mean Absolute Error')
        ax4.set_title('Error vs Computational Cost')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_control_variates_comparison(self, 
                                       cv_results: Dict[str, float],
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize control variates effectiveness
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Control Variates Analysis', fontsize=16, fontweight='bold')
        
        # 1. Variance comparison
        methods = ['Plain MC', 'Control Variates']
        variances = [cv_results['plain_variance'], cv_results['cv_variance']]
        colors = ['skyblue', 'orange']
        
        bars = ax1.bar(methods, variances, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Variance')
        ax1.set_title(f'Variance Comparison\n(Reduction: {cv_results["variance_reduction"]:.2f}x)')
        ax1.set_yscale('log')
        
        # Add value labels on bars
        for bar, var in zip(bars, variances):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                    f'{var:.2e}', ha='center', va='bottom', fontweight='bold')
        
        # 2. MSE comparison
        mses = [cv_results['plain_mse'], cv_results['cv_mse']]
        bars = ax2.bar(methods, mses, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Mean Squared Error')
        ax2.set_title(f'MSE Comparison\n(Reduction: {cv_results["mse_reduction"]:.2f}x)')
        ax2.set_yscale('log')
        
        for bar, mse in zip(bars, mses):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                    f'{mse:.2e}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Time comparison
        times = [cv_results['plain_time'], cv_results['cv_time']]
        bars = ax3.bar(methods, times, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Average Time (seconds)')
        ax3.set_title(f'Computational Time\n(Overhead: {(cv_results["time_overhead"]-1)*100:.1f}%)')
        
        for bar, t in zip(bars, times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                    f'{t:.4f}s', ha='center', va='bottom', fontweight='bold')
        
        # 4. Overall efficiency
        metrics = ['Variance\nReduction', 'MSE\nReduction', 'Time\nOverhead', 'Overall\nEfficiency']
        values = [cv_results['variance_reduction'], cv_results['mse_reduction'], 
                 cv_results['time_overhead'], cv_results['efficiency']]
        colors_metrics = ['green', 'blue', 'red', 'purple']
        
        bars = ax4.bar(metrics, values, color=colors_metrics, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Factor')
        ax4.set_title('Performance Metrics')
        ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Baseline')
        
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def run_comprehensive_analysis():
    """
    Run comprehensive convergence analysis suite
    """
    print("=" * 80)
    print("COMPREHENSIVE CONVERGENCE ANALYSIS")
    print("=" * 80)
    
    analyzer = ConvergenceAnalyzer(seed=42)
    
    # Parameters
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    
    # True prices (Black-Scholes)
    true_call = analyzer.engine.black_scholes_call(S0, K, T, r, sigma)
    true_put = analyzer.engine.black_scholes_put(S0, K, T, r, sigma)
    
    print(f"\nBenchmark Prices:")
    print(f"European Call: {true_call:.6f}")
    print(f"European Put:  {true_put:.6f}")
    
    # 1. European Call Convergence
    print(f"\n1. EUROPEAN CALL CONVERGENCE ANALYSIS")
    print("-" * 50)
    
    M_values = np.array([1000, 2500, 5000, 10000, 25000, 50000, 100000])
    
    convergence_results = analyzer.monte_carlo_convergence(
        analyzer.engine.european_call_mc,
        true_call,
        M_values,
        n_trials=20,
        S0=S0, K=K, T=T, r=r, sigma=sigma, antithetic=True
    )
    
    # Display results
    print(f"{'M':<8} {'Mean Price':<12} {'Std':<10} {'RMSE':<10} {'Time(s)':<8}")
    print("-" * 50)
    for i, M in enumerate(M_values):
        print(f"{M:<8} {convergence_results['mean_prices'][i]:<12.6f} "
              f"{convergence_results['std_prices'][i]:<10.6f} "
              f"{convergence_results['rmse'][i]:<10.6f} "
              f"{convergence_results['computation_times'][i]:<8.4f}")
    
    # Plot convergence
    fig1 = analyzer.plot_convergence(
        convergence_results, 
        true_call,
        "European Call Option - Monte Carlo Convergence Analysis",
        "european_call_convergence.png"
    )
    plt.show()
    
    # 2. Control Variates Analysis
    print(f"\n2. CONTROL VARIATES EFFECTIVENESS")
    print("-" * 50)
    
    cv_results = analyzer.control_variates_analysis(
        analyzer.engine.european_call_mc,
        true_call,
        M=50000,
        n_trials=30,
        S0=S0, K=K, T=T, r=r, sigma=sigma, antithetic=True
    )
    
    print(f"Variance Reduction:    {cv_results['variance_reduction']:.2f}x")
    print(f"MSE Reduction:         {cv_results['mse_reduction']:.2f}x")
    print(f"Time Overhead:         {(cv_results['time_overhead']-1)*100:.1f}%")
    print(f"Overall Efficiency:    {cv_results['efficiency']:.2f}x")
    
    fig2 = analyzer.plot_control_variates_comparison(
        cv_results,
        "control_variates_analysis.png"
    )
    plt.show()
    
    # 3. Asian Option Convergence
    print(f"\n3. ASIAN OPTION CONVERGENCE (EXOTIC)")
    print("-" * 50)
    
    M_values_asian = np.array([5000, 10000, 25000, 50000, 100000])
    
    # Get approximate benchmark (high-precision MC)
    asian_benchmark = analyzer.exotic_engine.asian_call_mc(
        S0, K, T, r, sigma, M=500000, antithetic=True
    )
    
    asian_convergence = analyzer.monte_carlo_convergence(
        analyzer.exotic_engine.asian_call_mc,
        asian_benchmark,
        M_values_asian,
        n_trials=15,
        S0=S0, K=K, T=T, r=r, sigma=sigma, antithetic=True
    )
    
    fig3 = analyzer.plot_convergence(
        asian_convergence,
        asian_benchmark,
        "Asian Call Option - Monte Carlo Convergence Analysis",
        "asian_call_convergence.png"
    )
    plt.show()
    
    # 4. Summary Report
    print(f"\n" + "=" * 80)
    print("CONVERGENCE ANALYSIS SUMMARY REPORT")
    print("=" * 80)
    
    # Calculate convergence rates
    log_M = np.log(M_values)
    log_rmse = np.log(convergence_results['rmse'])
    convergence_rate = -np.polyfit(log_M, log_rmse, 1)[0]
    
    print(f"\nEuropean Call Option:")
    print(f"  Theoretical Price:     {true_call:.6f}")
    print(f"  Final MC Estimate:     {convergence_results['mean_prices'][-1]:.6f}")
    print(f"  Final RMSE:            {convergence_results['rmse'][-1]:.6f}")
    print(f"  Convergence Rate:      O(M^{-convergence_rate:.3f}) [Theory: O(M^-0.5)]")
    print(f"  Computational Efficiency: {M_values[-1]/convergence_results['computation_times'][-1]:,.0f} sims/sec")
    
    print(f"\nControl Variates Performance:")
    print(f"  Variance Reduction:    {cv_results['variance_reduction']:.2f}x")
    print(f"  Computational Overhead: {(cv_results['time_overhead']-1)*100:.1f}%")
    print(f"  Net Efficiency Gain:   {cv_results['efficiency']:.2f}x")
    
    print(f"\nAsian Option (Exotic):")
    print(f"  Benchmark Price:       {asian_benchmark:.6f}")
    print(f"  Final RMSE:            {asian_convergence['rmse'][-1]:.6f}")
    
    print(f"\n🎯 ANALYSIS COMPLETE - Publication-ready results generated!")
    print(f"   Plots saved: european_call_convergence.png, control_variates_analysis.png, asian_call_convergence.png")


if __name__ == "__main__":
    run_comprehensive_analysis()
