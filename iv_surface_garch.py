import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from scipy.optimize import minimize
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

# Set dark blue and gold color scheme
plt.style.use('dark_background')
DARK_BLUE = '#0A1929'
GOLD = '#FFD700'
LIGHT_BLUE = '#1E88E5'
ORANGE = '#FFA726'


def fit_garch_model(returns, p=1, q=1):
    """Fit GARCH(p,q) model to return series"""
    try:
        if len(returns) < 50:
            return None
        
        # Fit GARCH model
        model = arch_model(returns, vol='Garch', p=p, q=q, rescale=False)
        result = model.fit(disp='off', show_warning=False)
        
        # Extract parameters
        params = result.params
        omega = params['omega']
        alpha = params['alpha[1]']
        beta = params['beta[1]']
        
        # Calculate persistence
        persistence = alpha + beta
        
        # Get conditional volatility
        cond_vol = result.conditional_volatility
        current_vol = float(cond_vol.iloc[-1])
        
        # Forecast volatility (30 days ahead for surface)
        forecasts = result.forecast(horizon=30)
        forecast_variance = forecasts.variance.values[-1, :]
        forecast_vol = np.sqrt(forecast_variance)
        
        # Calculate long-run volatility
        if persistence < 1:
            long_run_vol = np.sqrt(omega / (1 - persistence))
            half_life = np.log(0.5) / np.log(persistence) if persistence > 0 else 999
        else:
            long_run_vol = current_vol
            half_life = 999
        
        return {
            'omega': float(omega),
            'alpha': float(alpha),
            'beta': float(beta),
            'persistence': float(persistence),
            'current_vol': float(current_vol),
            'long_run_vol': float(long_run_vol),
            'conditional_volatility': cond_vol.tolist(),
            'forecast_vol': forecast_vol.tolist(),
            'is_stationary': persistence < 1,
            'half_life': float(half_life)
        }
        
    except Exception as e:
        print(f"GARCH fitting error: {e}")
        return None


def calculate_garch_volatility_regime(closes):
    """Enhanced volatility regime detection using GARCH"""
    # Calculate returns (in percentage)
    returns = np.log(closes[1:] / closes[:-1]) * 100
    
    # Fit GARCH model
    garch_results = fit_garch_model(returns)
    
    if garch_results is None:
        # Fallback to simple calculation if GARCH fails
        vol = np.std(returns) * np.sqrt(252)
        return {
            'regime': 'Normal Vol',
            'regime_factor': 1.0,
            'current_vol': float(vol),
            'long_run_vol': float(vol),
            'vol_ratio': 1.0,
            'vol_trend': 'Stable',
            'forecast_vol_5d': float(vol),
            'garch_params': None,
            'forecast_vol_array': [vol] * 30,
            'is_stationary': True
        }
    
    current_vol = garch_results['current_vol']
    long_run_vol = garch_results['long_run_vol']
    forecast_vol = garch_results['forecast_vol']
    
    # Calculate vol ratio (current vs long-run)
    vol_ratio = current_vol / long_run_vol if long_run_vol > 0 else 1.0
    
    # Determine regime based on GARCH parameters and current vol
    if vol_ratio > 1.5:
        regime = "Extreme Vol Spike"
        regime_factor = 1.8
    elif vol_ratio > 1.3:
        regime = "High Vol Spike"
        regime_factor = 1.5
    elif vol_ratio > 1.1:
        regime = "Elevated Vol"
        regime_factor = 1.2
    elif vol_ratio < 0.7:
        regime = "Extreme Vol Compression"
        regime_factor = 0.6
    elif vol_ratio < 0.85:
        regime = "Low Vol Compression"
        regime_factor = 0.75
    else:
        regime = "Normal Vol"
        regime_factor = 1.0
    
    # Calculate expected vol change (forward-looking)
    avg_forecast_vol = np.mean(forecast_vol[:5])  # Next 5 days
    vol_trend = "Increasing" if avg_forecast_vol > current_vol * 1.05 else \
                "Decreasing" if avg_forecast_vol < current_vol * 0.95 else \
                "Stable"
    
    return {
        'regime': regime,
        'regime_factor': regime_factor,
        'current_vol': float(current_vol),
        'long_run_vol': float(long_run_vol),
        'vol_ratio': float(vol_ratio),
        'vol_trend': vol_trend,
        'forecast_vol_5d': float(avg_forecast_vol),
        'forecast_vol_array': [float(v) for v in forecast_vol],
        'garch_params': {
            'omega': garch_results['omega'],
            'alpha': garch_results['alpha'],
            'beta': garch_results['beta'],
            'persistence': garch_results['persistence'],
            'half_life': garch_results['half_life']
        },
        'is_stationary': garch_results['is_stationary']
    }


def generate_volatility_surface(current_price, garch_vol_regime):
    """Generate implied volatility surface using GARCH-calibrated parameters"""
    
    if not garch_vol_regime or not isinstance(garch_vol_regime, dict):
        garch_vol_regime = {
            'garch_params': None,
            'forecast_vol_array': [],
            'current_vol': 20.0
        }
    
    if garch_vol_regime.get('garch_params') is not None:
        atm_vol = garch_vol_regime['current_vol'] / 100
    else:
        atm_vol = 0.20
    
    moneyness_range = np.linspace(0.7, 1.3, 25)  # More points for smoother surface
    strikes = [m * current_price for m in moneyness_range]
    maturities_days = [7, 14, 21, 30, 45, 60, 90, 120, 180, 365]
    maturities = [d / 365.0 for d in maturities_days]
    
    # Create meshgrid for 2D surface
    M, T = np.meshgrid(moneyness_range, maturities_days)
    IV_surface = np.zeros_like(M)
    
    for i, (T_days, T) in enumerate(zip(maturities_days, maturities)):
        for j, (moneyness, K) in enumerate(zip(moneyness_range, strikes)):
            # Volatility skew (negative for equity)
            skew = -0.15 * (moneyness - 1)
            # Volatility smile (convexity)
            smile = 0.08 * (moneyness - 1)**2
            # Term structure (increasing with time)
            term_structure = 0.03 * np.log(1 + T)
            
            # GARCH adjustment
            garch_adjustment = 0.0
            if garch_vol_regime.get('garch_params'):
                forecast_vols = garch_vol_regime.get('forecast_vol_array', [])
                if forecast_vols:
                    idx = min(T_days - 1, len(forecast_vols) - 1)
                    garch_adjustment = (forecast_vols[idx] / 100 - atm_vol) * 0.5
            
            iv = max(0.05, atm_vol + skew + smile + term_structure + garch_adjustment)
            IV_surface[i, j] = iv * 100  # Convert to percentage
    
    return {
        'moneyness': moneyness_range,
        'maturities_days': maturities_days,
        'maturities': maturities,
        'IV_surface': IV_surface,
        'current_price': float(current_price),
        'atm_vol': float(atm_vol * 100),
        'garch_calibrated': bool(garch_vol_regime.get('garch_params')),
        'garch_regime': garch_vol_regime
    }


def plot_2d_iv_surface(ticker='SPY', period='1y'):
    """Generate and plot 2D IV surface with GARCH calibration"""
    
    print(f"Fetching data for {ticker}...")
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    
    if len(hist) == 0:
        print(f"No data available for {ticker}")
        return
    
    closes = hist['Close'].values
    current_price = closes[-1]
    
    print("Fitting GARCH model...")
    garch_vol_regime = calculate_garch_volatility_regime(closes)
    print(f"GARCH Regime: {garch_vol_regime['regime']}")
    print(f"Current Vol: {garch_vol_regime['current_vol']:.2f}%")
    print(f"Persistence: {garch_vol_regime['garch_params']['persistence']:.3f}" if garch_vol_regime['garch_params'] else "N/A")
    
    print("Generating volatility surface...")
    surface_data = generate_volatility_surface(current_price, garch_vol_regime)
    
    # Create figure with dark background
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(DARK_BLUE)
    
    # 2D Contour Plot
    ax1 = plt.subplot(2, 2, 1)
    M, T = np.meshgrid(surface_data['moneyness'], surface_data['maturities_days'])
    contour = ax1.contourf(M, T, surface_data['IV_surface'], levels=20, cmap='viridis', alpha=0.9)
    ax1.contour(M, T, surface_data['IV_surface'], levels=20, colors='white', alpha=0.3, linewidths=0.5)
    ax1.set_xlabel('Moneyness (K/S)', color=GOLD, fontsize=11)
    ax1.set_ylabel('Maturity (Days)', color=GOLD, fontsize=11)
    ax1.set_title('Implied Volatility Surface (2D Contour)', color=GOLD, fontsize=13, fontweight='bold')
    ax1.tick_params(colors=LIGHT_BLUE)
    ax1.grid(True, alpha=0.2, color=LIGHT_BLUE)
    ax1.set_facecolor(DARK_BLUE)
    cbar1 = plt.colorbar(contour, ax=ax1)
    cbar1.set_label('Implied Volatility (%)', color=GOLD, fontsize=10)
    cbar1.ax.tick_params(colors=LIGHT_BLUE)
    
    # 2D Heatmap
    ax2 = plt.subplot(2, 2, 2)
    im = ax2.imshow(surface_data['IV_surface'], aspect='auto', origin='lower', 
                    extent=[surface_data['moneyness'].min(), surface_data['moneyness'].max(),
                           surface_data['maturities_days'].min(), surface_data['maturities_days'].max()],
                    cmap='viridis', interpolation='bilinear')
    ax2.set_xlabel('Moneyness (K/S)', color=GOLD, fontsize=11)
    ax2.set_ylabel('Maturity (Days)', color=GOLD, fontsize=11)
    ax2.set_title('Implied Volatility Surface (2D Heatmap)', color=GOLD, fontsize=13, fontweight='bold')
    ax2.tick_params(colors=LIGHT_BLUE)
    ax2.set_facecolor(DARK_BLUE)
    cbar2 = plt.colorbar(im, ax=ax2)
    cbar2.set_label('Implied Volatility (%)', color=GOLD, fontsize=10)
    cbar2.ax.tick_params(colors=LIGHT_BLUE)
    
    # Volatility Smile (IV vs Moneyness) for different maturities
    ax3 = plt.subplot(2, 2, 3)
    selected_maturities = [7, 30, 90, 365]
    for T_days in selected_maturities:
        if T_days in surface_data['maturities_days']:
            idx = surface_data['maturities_days'].index(T_days)
            ax3.plot(surface_data['moneyness'], surface_data['IV_surface'][idx, :], 
                    marker='o', markersize=4, linewidth=2, label=f'{T_days}d', alpha=0.8)
    ax3.set_xlabel('Moneyness (K/S)', color=GOLD, fontsize=11)
    ax3.set_ylabel('Implied Volatility (%)', color=GOLD, fontsize=11)
    ax3.set_title('Volatility Smile by Maturity', color=GOLD, fontsize=13, fontweight='bold')
    ax3.tick_params(colors=LIGHT_BLUE)
    ax3.grid(True, alpha=0.2, color=LIGHT_BLUE)
    ax3.set_facecolor(DARK_BLUE)
    ax3.legend(loc='best', facecolor=DARK_BLUE, edgecolor=GOLD, labelcolor=LIGHT_BLUE)
    ax3.axvline(x=1.0, color=GOLD, linestyle='--', alpha=0.5, linewidth=1)
    
    # Term Structure (IV vs Maturity) for different moneyness levels
    ax4 = plt.subplot(2, 2, 4)
    selected_moneyness = [0.8, 0.9, 1.0, 1.1, 1.2]
    for moneyness in selected_moneyness:
        if moneyness in surface_data['moneyness']:
            idx = np.argmin(np.abs(surface_data['moneyness'] - moneyness))
            ax4.plot(surface_data['maturities_days'], surface_data['IV_surface'][:, idx], 
                    marker='s', markersize=4, linewidth=2, label=f'K/S={moneyness:.2f}', alpha=0.8)
    ax4.set_xlabel('Maturity (Days)', color=GOLD, fontsize=11)
    ax4.set_ylabel('Implied Volatility (%)', color=GOLD, fontsize=11)
    ax4.set_title('Volatility Term Structure by Moneyness', color=GOLD, fontsize=13, fontweight='bold')
    ax4.tick_params(colors=LIGHT_BLUE)
    ax4.grid(True, alpha=0.2, color=LIGHT_BLUE)
    ax4.set_facecolor(DARK_BLUE)
    ax4.legend(loc='best', facecolor=DARK_BLUE, edgecolor=GOLD, labelcolor=LIGHT_BLUE)
    
    # Add title with GARCH info
    garch_status = "GARCH-Calibrated" if surface_data['garch_calibrated'] else "No GARCH"
    regime = surface_data['garch_regime']['regime']
    fig.suptitle(f'{ticker} Implied Volatility Surface - {garch_status} | Regime: {regime} | Current Price: ${current_price:.2f}',
                color=GOLD, fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f'{ticker}_iv_surface_2d.png', dpi=300, facecolor=DARK_BLUE, bbox_inches='tight')
    print(f"\nSaved: {ticker}_iv_surface_2d.png")
    plt.show()


def plot_3d_iv_surface(ticker='SPY', period='1y'):
    """Generate and plot 3D IV surface with GARCH calibration"""
    
    print(f"Fetching data for {ticker}...")
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    
    if len(hist) == 0:
        print(f"No data available for {ticker}")
        return
    
    closes = hist['Close'].values
    current_price = closes[-1]
    
    print("Fitting GARCH model...")
    garch_vol_regime = calculate_garch_volatility_regime(closes)
    
    print("Generating volatility surface...")
    surface_data = generate_volatility_surface(current_price, garch_vol_regime)
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(DARK_BLUE)
    ax = fig.add_subplot(111, projection='3d')
    
    M, T = np.meshgrid(surface_data['moneyness'], surface_data['maturities_days'])
    
    # Create surface plot
    surf = ax.plot_surface(M, T, surface_data['IV_surface'], 
                          cmap='viridis', alpha=0.9, 
                          linewidth=0, antialiased=True,
                          edgecolor='none')
    
    ax.set_xlabel('Moneyness (K/S)', color=GOLD, fontsize=11, labelpad=10)
    ax.set_ylabel('Maturity (Days)', color=GOLD, fontsize=11, labelpad=10)
    ax.set_zlabel('Implied Volatility (%)', color=GOLD, fontsize=11, labelpad=10)
    
    ax.tick_params(colors=LIGHT_BLUE)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(DARK_BLUE)
    ax.yaxis.pane.set_edgecolor(DARK_BLUE)
    ax.zaxis.pane.set_edgecolor(DARK_BLUE)
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, pad=0.1)
    cbar.set_label('Implied Volatility (%)', color=GOLD, fontsize=10)
    cbar.ax.tick_params(colors=LIGHT_BLUE)
    
    garch_status = "GARCH-Calibrated" if surface_data['garch_calibrated'] else "No GARCH"
    regime = surface_data['garch_regime']['regime']
    ax.set_title(f'{ticker} 3D IV Surface - {garch_status} | Regime: {regime} | Price: ${current_price:.2f}',
                color=GOLD, fontsize=13, fontweight='bold', pad=20)
    
    plt.savefig(f'{ticker}_iv_surface_3d.png', dpi=300, facecolor=DARK_BLUE, bbox_inches='tight')
    print(f"\nSaved: {ticker}_iv_surface_3d.png")
    plt.show()


if __name__ == "__main__":
    # Example usage
    ticker = 'SPY'  # Change to any ticker
    
    print("="*60)
    print("GARCH-Calibrated Implied Volatility Surface Visualization")
    print("="*60)
    
    # Plot 2D surface (contour + heatmap + slices)
    plot_2d_iv_surface(ticker=ticker, period='1y')
    
    # Plot 3D surface
    # plot_3d_iv_surface(ticker=ticker, period='1y')



