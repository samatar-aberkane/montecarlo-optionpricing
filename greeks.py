# calculs de Delta/Vega

from pricing import european_call_mc, european_put_mc, black_scholes_call, black_scholes_put

# -------------------------
# Delta
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

# -------------------------
# Gamma
# -------------------------
def gamma_mc(option_type, S0, K, T, r, sigma, eps=1e-4, **kwargs):
    """Approximate Gamma via finite differences"""
    if option_type.lower() == 'call':
        price_up = european_call_mc(S0 + eps, K, T, r, sigma, **kwargs)
        price = european_call_mc(S0, K, T, r, sigma, **kwargs)
        price_down = european_call_mc(S0 - eps, K, T, r, sigma, **kwargs)
    elif option_type.lower() == 'put':
        price_up = european_put_mc(S0 + eps, K, T, r, sigma, **kwargs)
        price = european_put_mc(S0, K, T, r, sigma, **kwargs)
        price_down = european_put_mc(S0 - eps, K, T, r, sigma, **kwargs)
    else:
        raise ValueError("Option type must be 'call' or 'put'")
    return (price_up - 2*price + price_down) / (eps**2)

# -------------------------
# Vega
# -------------------------
def vega_mc(option_type, S0, K, T, r, sigma, eps=1e-4, **kwargs):
    """Approximate Vega via finite differences"""
    if option_type.lower() == 'call':
        price_up = european_call_mc(S0, K, T, r, sigma + eps, **kwargs)
        price_down = european_call_mc(S0, K, T, r, sigma - eps, **kwargs)
    elif option_type.lower() == 'put':
        price_up = european_put_mc(S0, K, T, r, sigma + eps, **kwargs)
        price_down = european_put_mc(S0, K, T, r, sigma - eps, **kwargs)
    else:
        raise ValueError("Option type must be 'call' or 'put'")
    return (price_up - price_down) / (2 * eps)

# -------------------------
# Theta
# -------------------------
def theta_mc(option_type, S0, K, T, r, sigma, eps=1e-4, **kwargs):
    """Approximate Theta via finite differences (per year)"""
    if option_type.lower() == 'call':
        price_up = european_call_mc(S0, K, T + eps, r, sigma, **kwargs)
        price_down = european_call_mc(S0, K, T - eps, r, sigma, **kwargs)
    elif option_type.lower() == 'put':
        price_up = european_put_mc(S0, K, T + eps, r, sigma, **kwargs)
        price_down = european_put_mc(S0, K, T - eps, r, sigma, **kwargs)
    else:
        raise ValueError("Option type must be 'call' or 'put'")
    return (price_down - price_up) / (2 * eps)

# -------------------------
# Rho
# -------------------------
def rho_mc(option_type, S0, K, T, r, sigma, eps=1e-4, **kwargs):
    """Approximate Rho via finite differences"""
    if option_type.lower() == 'call':
        price_up = european_call_mc(S0, K, T, r + eps, sigma, **kwargs)
        price_down = european_call_mc(S0, K, T, r - eps, sigma, **kwargs)
    elif option_type.lower() == 'put':
        price_up = european_put_mc(S0, K, T, r + eps, sigma, **kwargs)
        price_down = european_put_mc(S0, K, T, r - eps, sigma, **kwargs)
    else:
        raise ValueError("Option type must be 'call' or 'put'")
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

    print("Delta Call =", delta_mc('call', S0, K, T, r, sigma, M=50000, antithetic=True))
    print("Gamma Call =", gamma_mc('call', S0, K, T, r, sigma, M=50000, antithetic=True))
    print("Vega Call =", vega_mc('call', S0, K, T, r, sigma, M=50000, antithetic=True))
    print("Theta Call =", theta_mc('call', S0, K, T, r, sigma, M=50000, antithetic=True))
    print("Rho Call =", rho_mc('call', S0, K, T, r, sigma, M=50000, antithetic=True))
