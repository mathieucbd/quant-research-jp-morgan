"""
Commodities analysis and forecasting script
-------------------------------------------
This script loads historical natural gas price data, fits forecasting models
(Holt-Winters), exports forecasts, and creates a daily interpolated price series.

Usage:
    from commodities_analysis import estimate_price, build_daily_series
    daily_prices = build_daily_series(df, forecast)
    price = estimate_price('2025-01-15', daily_prices)
"""

import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing

# ------------------------------
# Load and prepare data
# ------------------------------
def load_nat_gas_data(path: str = "data/Nat_Gas.csv") -> pd.DataFrame:
    """Load and prepare natural gas price dataset."""
    df = pd.read_csv(path)
    df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
    df = df.sort_values('Dates').set_index('Dates')
    return df

# ------------------------------
# Forecasting
# ------------------------------
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

def forecast_future_prices(df: pd.DataFrame, months_ahead: int = 12) -> pd.Series:
    """Forecast future natural gas prices using Holt-Winters."""
    fit_hw = fit_holt_winters(df['Prices'])
    forecast = fit_hw.forecast(months_ahead)
    return forecast

# ------------------------------
# Daily interpolation and estimate function
# ------------------------------
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

def estimate_price(date: str, daily: pd.Series) -> float:
    """Return interpolated natural gas price (USD/MMBtu) for a given date."""
    d = pd.to_datetime(date)
    return float(daily.loc[d])

# ------------------------------
# Main execution
# ------------------------------
if __name__ == "__main__":
    EXPORT_DAILY = False 

    df = load_nat_gas_data()
    forecast_hw = forecast_future_prices(df)
    daily = build_daily_series(df, forecast_hw)

    if EXPORT_DAILY:
        daily_df = daily.reset_index()
        daily_df.columns = ['Dates', 'Prices']
        daily_df.to_csv("data/Nat_Gas_Daily.csv", index=False)
        print("✅ Daily series exported to data/Nat_Gas_Daily.csv")

    # Example usage
    example_date = "2025-01-15"
    print(f"Estimated price for {example_date}: ${estimate_price(example_date, daily):.2f}")