import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime
import scipy.optimize as opt
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from scipy.stats import norm
import torch
import math

# this function takes the ticker as input and grabs the first 12 available maturities
def fetch_raw_volatility_surface(ticker_symbol="SPY", max_expirations=12):
    ticker = yf.Ticker(ticker_symbol)

    # grabbing the latest available close price for the underlying
    spot_price = ticker.history(period="1d")['Close'].iloc[-1]
    
    expirations = ticker.options[:max_expirations]
    options_data = []
    today = datetime.today()

    #loop through expirations to build a dataframe organized by time to maturity
    for exp in expirations:
        opt_chain = ticker.option_chain(exp)

        exp_date = datetime.strptime(exp, '%Y-%m-%d')
        ttm = (exp_date - today).days / 365.0
        
        # skiping options that expire too soon
        if ttm <= 0.01:
            continue

        calls = opt_chain.calls.copy() #creating a copy 
        calls = calls[calls['strike'] > spot_price] # we only use OTM call options
        calls['type'] = 'Call' # We assign to all of them the label 'Call'

        puts = opt_chain.puts.copy()
        puts = puts[puts['strike'] < spot_price] 
        puts['type'] = 'Put'

        # merge the two dataframes with new indexes
        chain = pd.concat([calls, puts], ignore_index=True)
        chain['TTM'] = ttm # attach the calculated TTM to this batch
        options_data.append(chain) 

    df = pd.concat(options_data, ignore_index=True) 
    
    # filter  high or low implied volatilities to clean the data
    df_clean = df[(df['impliedVolatility'] > 0.01) & (df['impliedVolatility'] < 2.5)].copy()

    
    df_clean['moneyness'] = df_clean['strike'] / spot_price
    df_clean['log_moneyness'] = np.log(df_clean['moneyness'])
    df_clean['total_variance'] = (df_clean['impliedVolatility'] ** 2) * df_clean['TTM']

   
    return df_clean, spot_price

#SVI formula to calculate variance
def svi_variance(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

#loss function 
def objective_function(params, k_data, w_data):
    a, b, rho, m, sigma = params
    w_svi = svi_variance(k_data, a, b, rho, m, sigma)
    
    # We strongly penalize if the model returns negative variance
    if np.any(w_svi < 0):
        return 1e6
        
    mse = np.sum((w_svi - w_data)**2)
    return mse


# doing this globally so the fast heston pricer doesn't recalculate it every time
nodes, weights = np.polynomial.legendre.leggauss(100)
A = 100.0  
mapped_nodes = 0.5 * A * nodes + 0.5 * A
mapped_weights = 0.5 * A * weights

# computes the heston call price using fourier inversion
# it's faster because of the pre-computed legendre nodes above
def heston_call_price_fast(S0, K, T, r, kappa, theta, xi, rho, v0):
    i = 1j
    u = mapped_nodes
    alpha = 1.5  

    # building the characteristic function 
    phi = u - (alpha + 1) * i
    d = np.sqrt((rho * xi * phi * i - kappa)**2 + xi**2 * (phi * i + phi**2))
    g = (kappa - rho * xi * phi * i - d) / (kappa - rho * xi * phi * i + d)

    C = r * phi * i * T + (kappa * theta / xi**2) * \
        ((kappa - rho * xi * phi * i - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
    D = ((kappa - rho * xi * phi * i - d) / xi**2) * \
        ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))

    cf = np.exp(C + D * v0 + i * phi * np.log(S0))
    denominator = (alpha**2 + alpha - u**2) + i * (2 * alpha + 1) * u

    psi = np.exp(-r * T) * cf / denominator
    integrand_values = (np.exp(-i * u * np.log(K)) * psi).real
    integral = np.sum(integrand_values * mapped_weights)

    return np.exp(-alpha * np.log(K)) / np.pi * integral

# classic black-scholes formula 
def bs_call_price(S0, K, T, r, iv):
    d1 = (np.log(S0 / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# function to find how far our heston prices are from the market targets
def heston_objective(params, market_data, S0, r):
    kappa, theta, xi, rho, v0 = params
    mse = 0.0
    for K, T, svi_iv in market_data:
        target_price = bs_call_price(S0, K, T, r, svi_iv)
        heston_price = heston_call_price_fast(S0, K, T, r, kappa, theta, xi, rho, v0)

        
        if np.isnan(heston_price) or np.isinf(heston_price):
            return 1e10

        error = (heston_price - target_price) ** 2
        mse += error
    return mse / len(market_data)

# just a step for the full truncation scheme to prevent negative variances
def heston_full_truncation_step(v_t, kappa, theta, xi, dt, Z_v):
    v_t_plus = np.maximum(v_t, 0.0)
    drift = kappa * (theta - v_t_plus) * dt
    diffusion = xi * np.sqrt(v_t_plus) * np.sqrt(dt) * Z_v
    return v_t + drift + diffusion

def calibrate_heston(market_data, S0, r):
    
    #limits for the heston parameters
    bounds = [
        (0.1, 5.0),    # kappa
        (0.01, 0.5),   # theta
        (0.01, 1.5),   # xi
        (-0.99, 0.0),  # rho
        (0.005, 0.2)   # v0
    ]

    # using differential evolution to search for the best parameters
    result = opt.differential_evolution(
        heston_objective,
        bounds=bounds,
        args=(market_data, S0, r),
        strategy='best1bin',
        maxiter=100,         
        popsize=8,
        tol=1e-5,
        workers=-1,
        updating='deferred',
        polish=False,        
        disp=True
    )

    kappa, theta, xi, rho, v0 = result.x
    print(f"Kappa (Mean Reversion Speed) : {kappa:.4f}")
    print(f"Theta (Long-term Variance)   : {theta:.4f}")
    print(f"Xi    (Vol of Vol)           : {xi:.4f}")
    print(f"Rho   (Correlation)          : {rho:.4f}")
    print(f"v0    (Initial Variance)     : {v0:.4f}")

    # checking if the feller condition holds
    feller = 2 * kappa * theta - xi**2
    if feller > 0:
        pass
    else:
        print(f"\nFeller condition violated (2*k*theta - xi^2 = {feller:.4f} < 0)")

    return result.x





if __name__ == '__main__':
    
    # making sure plots open in the browser
    pio.renderers.default = 'browser'

    
    df_surface, spot = fetch_raw_volatility_surface("SPY")

    # plotting the raw volatility data as a 3d scatter plot
    fig1 = go.Figure(data=[go.Scatter3d(
        x=df_surface['log_moneyness'],
        y=df_surface['TTM'],
        z=df_surface['impliedVolatility'],
        mode='markers',
        marker=dict(size=4, color=df_surface['impliedVolatility'], colorscale='Viridis', opacity=0.8),
        text=df_surface['type'] + " | Strike: $" + df_surface['strike'].astype(str)
    )])
    fig1.update_layout(
        title=f"Volatility Surface Grezza (SPY) - Prezzo Spot: ${spot:.2f}",
        scene=dict(xaxis_title='Log-Moneyness ln(K/S)', yaxis_title='Time to Maturity (Anni)', zaxis_title='Implied Volatility (%)'),
        width=900, height=700, margin=dict(l=0, r=0, b=0, t=40)
    )
    fig1.show()

    scadenze_uniche = sorted(df_surface['TTM'].unique())
    svi_parameters = {}

    
    #Looping through each maturity slice to calibrate SVI
    for ttm in scadenze_uniche:
        df_slice = df_surface[df_surface['TTM'] == ttm]

        # skip if we don't have enough data points for a good fit
        if len(df_slice) < 5:
            print(f"  - TTM {ttm:.3f}: too few data)")
            continue

        k_market = df_slice['log_moneyness'].values
        w_market = df_slice['total_variance'].values

        bounds = [
            (min(w_market)*0.8, max(w_market)),
            (0.01, 2.0),
            (-0.99, 0.99),
            (min(k_market), max(k_market)),
            (0.001, 1.0)
        ]

        res = opt.differential_evolution(
            objective_function,
            bounds=bounds,
            args=(k_market, w_market),
            strategy='best1bin',
            maxiter=1000,
            popsize=15,
            tol=1e-5,
            seed=42
        )

        if res.success:
            svi_parameters[ttm] = res.x
            pass
        else:
            print(f"  - TTM {ttm:.3f}: Calibration failed")

    valid_ttms = list(svi_parameters.keys())
    
    if len(valid_ttms) < 2:
        print("\nERROR: Not enough ttms to build the surface.")
    else:
        # interpolating the params so we can generate points between our known maturities
        params_array = np.array(list(svi_parameters.values()))
        param_interpolator = interp1d(valid_ttms, params_array, axis=0, kind='linear', fill_value='extrapolate')

        k_grid = np.linspace(df_surface['log_moneyness'].min(), df_surface['log_moneyness'].max(), 60)
        t_grid = np.linspace(min(valid_ttms), max(valid_ttms), 60)
        K_mesh, T_mesh = np.meshgrid(k_grid, t_grid)
        IV_surface_mesh = np.zeros_like(K_mesh)

        # calculating implied vol for every point on the grid
        for i in range(K_mesh.shape[0]):
            for j in range(K_mesh.shape[1]):
                t_val = T_mesh[i, j]
                k_val = K_mesh[i, j]
                p = param_interpolator(t_val)
                w_val = svi_variance(k_val, *p)
                IV_surface_mesh[i, j] = np.sqrt(max(w_val, 1e-6) / t_val)

        fig2 = go.Figure()
        fig2.add_trace(go.Surface(x=K_mesh, y=T_mesh, z=IV_surface_mesh, colorscale='Plasma', opacity=0.7, name='SVI Surface'))
        fig2.add_trace(go.Scatter3d(
            x=df_surface['log_moneyness'], y=df_surface['TTM'], z=df_surface['impliedVolatility'],
            mode='markers', marker=dict(size=4, color='black', opacity=0.9), name='Dati Grezzi Mercato'
        ))
        fig2.update_layout(
            title="Volatility Surface (Differential Evolution)",
            scene=dict(xaxis_title='Log-Moneyness ln(K/S)', yaxis_title='Time to Maturity (Anni)', zaxis_title='Implied Volatility (%)'),
            width=1000, height=800, margin=dict(l=0, r=0, b=0, t=40)
        )
        fig2.show()



        r = 0.04 
        market_data_sampled = []
        k_sample = np.linspace(-0.2, 0.2, 7) 

        
        # grabbing points from our SVI surface to feed into Heston
        for t_val in valid_ttms:
            if t_val < 0.05:
                continue
            p = param_interpolator(t_val)
            for k_val in k_sample:
                K_val_sample = spot * np.exp(k_val)
                w_val = svi_variance(k_val, *p)
                svi_iv = np.sqrt(max(w_val, 1e-6) / t_val)
                market_data_sampled.append((K_val_sample, t_val, svi_iv))

        
        heston_params = calibrate_heston(market_data_sampled, spot, r)


        
        if heston_params is not None:
            kappa_opt, theta_opt, xi_opt, rho_opt, v0_opt = heston_params

            K_test = spot  
            T_test = 1.0         
            num_steps = 252
            num_paths = 100000

            

            # wrapping inputs in PyTorch tensors and turning on requires_grad 
            # so the engine tracks everything for the greeks later
            S0_tensor = torch.tensor(spot, requires_grad=True)
            v0_tensor = torch.tensor(v0_opt, requires_grad=True)
            K_tensor = torch.tensor(K_test)
            T_tensor = torch.tensor(T_test)
            r_tensor = torch.tensor(r)

            dt = T_tensor / num_steps
            half_paths = int(num_paths / 2)
            
            # generating random normal variables
            Z1_half = torch.randn(num_steps, half_paths)
            Z2_half = torch.randn(num_steps, half_paths)

            # using antithetic variates (adding the negative paths) to reduce variance in our simulation
            Z1 = torch.cat((Z1_half, -Z1_half), dim=1)
            Z2 = torch.cat((Z2_half, -Z2_half), dim=1)

            # correlating the brownian motions
            Z_S = Z1
            Z_v = rho_opt * Z1 + math.sqrt(1.0 - rho_opt**2) * Z2

            S_t = S0_tensor * torch.ones(num_paths)
            v_t = v0_tensor * torch.ones(num_paths)

            for t in range(num_steps):
                # gradient breaker: if we just use max(0, v_t), PyTorch gradients might explode or go NaN.
                #clamping it slightly above zero fixes this.
                v_plus = torch.clamp(v_t, min=1e-4)

                #variance step
                drift_v = kappa_opt * (theta_opt - v_plus) * dt
                diff_v = xi_opt * torch.sqrt(v_plus) * math.sqrt(dt) * Z_v[t]
                v_next = v_t + drift_v + diff_v

                #price step
                esp_S = (r_tensor - 0.5 * v_plus) * dt + torch.sqrt(v_plus) * math.sqrt(dt) * Z_S[t]
                S_next = S_t * torch.exp(esp_S)

                v_t = v_next
                S_t = S_next

            # calculating final payoff and discounting it back to present
            payoff = torch.max(S_t - K_tensor, torch.tensor(0.0))
            estimated_price = torch.exp(-r_tensor * T_tensor) * torch.mean(payoff)

            #with '.backward' we compute all the greeks using auto-differentiation
            estimated_price.backward()

            # extracting delta and vega 
            delta = S0_tensor.grad.item()
            vega = v0_tensor.grad.item()

            print(f"Call price (ATM, 1Y) : {estimated_price.item():.4f}")
            print(f"Delta (Δ)            : {delta:.4f}")
            print(f"Vega                 : {vega:.4f}")