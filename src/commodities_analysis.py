"""
Commodities analysis and forecasting script
-------------------------------------------
This script loads historical natural gas price data, fits forecasting models
(Holt-Winters and SARIMAX), exports forecasts, and creates a daily
interpolated price series with an `estimate_price(date)` function.

Usage:
    from commodities_analysis import estimate_price
    price = estimate_price('2025-01-15')
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from datetime import datetime

# ==============================
# Load and prepare data
# ==============================

def load_nat_gas_data(path: str = "data/Nat_Gas.csv") -> pd.DataFrame:
    """Load and prepare the natural gas price dataset."""
    df = pd.read_csv(path)
    df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
    df = df.sort_values('Dates').set_index('Dates')
    return df


# ==============================
# Forecasting model
# ==============================

def fit_holt_winters(train: pd.Series, seasonal_periods: int = 12):
    """Fit Holt-Winters Exponential Smoothing model."""
    model = ExponentialSmoothing(
        train,
        seasonal='add',
        trend='add',
        seasonal_periods=seasonal_periods,
        freq='ME'
    )
    fit = model.fit()
    return fit


# ==============================
# Forecast extrapolation
# ==============================

def forecast_future_prices(df: pd.DataFrame, months_ahead: int = 12) -> pd.Series:
    """Forecast future natural gas prices using Holt-Winters."""
    fit_hw = fit_holt_winters(df['Prices'])
    forecast = fit_hw.forecast(months_ahead)
    return forecast


# ==============================
# Daily interpolation and estimate function
# ==============================

def build_daily_series(df: pd.DataFrame, forecast: pd.Series) -> pd.Series:
    """Combine observed and forecasted prices, interpolate to daily frequency."""
    combined = pd.concat([df['Prices'], forecast])
    daily_index = pd.date_range(start=combined.index[0], end=combined.index[-1], freq='D')
    daily = (
        combined
        .reindex(combined.index.union(daily_index))
        .interpolate(method='cubic')
        .reindex(daily_index)
    )
    return daily


def estimate_price(date: str) -> float:
    """Return interpolated natural gas price (USD/MMBtu) for a given date."""
    global daily
    d = pd.to_datetime(date)
    return float(daily.loc[d])


# ==============================
# Main execution (if run directly)
# ==============================

if __name__ == "__main__":
    # Load data
    df = load_nat_gas_data()

    # Split for evaluation
    train, test = df.iloc[:-12], df.iloc[-12:]

    # Forecast next 12 months
    forecast_hw = forecast_future_prices(df)

    # Export forecast to CSV
    df_forecast = forecast_hw.reset_index()
    df_forecast.columns = ['Dates', 'Prices']
    df_forecast.to_csv('data/Nat_Gas_Forecast.csv', index=False)
    print("Forecast exported to data/Nat_Gas_Forecast.csv")

    # Build daily interpolation
    daily = build_daily_series(df, forecast_hw)

    # Example usage
    example_date = "2025-01-15"
    print(f"Estimated price for {example_date}: ${estimate_price(example_date):.2f} per MMBtu")
