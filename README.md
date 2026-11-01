# Quantitative Pricing Architecture 

This repository contains a quantitative finance architecture designed for robust option pricing, market data processing, and Greeks extraction.

## Market Data & Volatility Surface
**Objective:** Process raw market data to construct a continuous, stable volatility surface free of static arbitrages.
*   **Raw Data Extraction:** Fetches real-time option chains (strikes, expirations) from Yahoo Finance (e.g., SPY) and cleans the data by removing outliers and filtering by Implied Volatility (IV).
*   **SVI Parametrization:** Implements Gatheral's Stochastic Volatility Inspired (SVI) model to fit the implied variance slices. This ensures the generated surface is free from calendar and butterfly arbitrages.
*   **Surface Generation:** Interpolates SVI parameters across maturities to build a dense, continuous volatility surface and visualizes it using interactive 3D plots.

## Stochastic Volatility Model
**Objective:** Calibrate the Heston stochastic volatility model to the SVI surface by minimizing the Mean Squared Error, while strictly managing the stability of the variance process.
*   **Fourier Pricing:** Implements a fast Fourier inversion algorithm for Heston call pricing, utilizing pre-computed Gauss-Legendre quadrature nodes and weights for maximum efficiency.
*   **Robust Calibration:** Uses the Differential Evolution algorithm (`scipy.optimize`) to search the parameter space (κ, θ, ξ, ρ, v₀).
*   **Feller Condition Check:** Actively monitors the Feller condition (2κθ > ξ²) to ensure the strict positivity of the variance process.

## Monte Carlo Pricing & Greeks 
**Objective:** Develop a vectorized Monte Carlo pricing engine with advanced variance reduction techniques and exact sensitivity extraction.
*   **Vectorized Simulation:** Simulates the asset price and variance paths using PyTorch tensors.
*   **Variance Reduction:** Integrates Antithetic Variates to significantly reduce standard error and convergence time.
*   **Algorithmic Differentiation (AAD):** Abandons standard finite-difference bumps. Instead, it leverages PyTorch's `Autograd` to extract exact, analytical Greeks (Delta Δ and Vega V) directly from the simulated paths, guaranteeing hedging precision.


### Repository Contents
*   `pricing.py`: The complete Python script containing Market Data & Volatility Surface, Stochastic Volatility Model and Monte Carlo Pricing & Greeks.
