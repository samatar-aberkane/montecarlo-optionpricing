# Monte Carlo Option Pricing Project

This project explores the valuation of financial options using the Monte Carlo simulation method. It includes the pricing of European and exotic options, variance reduction techniques, and the calculation of "Greeks."

## Project Structure

* `pricing.py`: Contains the core functions for **Geometric Brownian Motion** (GBM) simulation and the pricing of **European options** (calls and puts). It also includes the analytical **Black-Scholes** formulas for comparison.
* `exotic_options.py`: Extends the core functionality to **exotic options**, including Asian options (based on the arithmetic average) and Barrier (Up-and-Out) options.
* `greeks.py`: Provides methods for calculating the **Greeks** (Delta, Gamma, Vega, Theta, Rho) using the **finite difference** method.
* `demo.ipynb`: A Jupyter notebook that demonstrates the use of the different modules. It generates convergence plots, payoff histograms, and displays calculated prices and Greeks.

## Key Features

* **GBM Simulation**: Models the price evolution of an underlying asset over time.
* **Option Pricing**: Values European, Asian, and Barrier options.
* **Variance Reduction**: Employs **antithetic variates** to improve the efficiency and accuracy of the simulation.
* **Greeks Calculation**: Derives key sensitivities of the option price with respect to market parameters.
* **Visual Analysis**: The demo notebook allows for the visualization of simulation convergence and payoff distribution.
