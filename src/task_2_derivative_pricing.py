"""
Derivative pricing script for gas storage contracts
---------------------------------------------------
Uses historical + forecasted daily prices to calculate fair contract value.
"""

import pandas as pd
from task_1_commodities_analysis import estimate_price

def price_gas_storage_contract(
    price_data: pd.DataFrame,       # DataFrame with 'Prices' column and datetime index
    injection_dates: list,          # list of YYYY-MM-DD dates
    withdrawal_dates: list,         # list of YYYY-MM-DD dates
    injection_rate: float,          # MMBtu per injection month
    withdrawal_rate: float,         # MMBtu per withdrawal month
    max_storage: float,             # MMBtu (max capacity)
    storage_fee_per_month: float,   # USD/month
    inject_withdraw_fee: float,     # USD/MMBtu (variable)
    transport_fee: float            # USD per injection/withdrawal event
) -> float:
    """
    Calculate fair value (USD) of a natural gas storage contract.
    """
    total_value = 0.0
    stored_volume = 0.0

    injection_dates = [pd.Timestamp(d) for d in injection_dates]
    withdrawal_dates = [pd.Timestamp(d) for d in withdrawal_dates]

    # --- Injection ---
    for date in injection_dates:
        if date not in price_data.index:
            raise ValueError(f"Injection date {date} not in price data.")
        price = price_data.loc[date, 'Prices']
        volume = min(injection_rate, max_storage - stored_volume)
        stored_volume += volume
        total_value -= price * volume
        total_value -= inject_withdraw_fee * volume + transport_fee

    # --- Withdrawal ---
    for date in withdrawal_dates:
        if date not in price_data.index:
            raise ValueError(f"Withdrawal date {date} not in price data.")
        price = price_data.loc[date, 'Prices']
        volume = min(withdrawal_rate, stored_volume)
        stored_volume -= volume
        total_value += price * volume
        total_value -= inject_withdraw_fee * volume + transport_fee

    # --- Storage fees ---
    if injection_dates and withdrawal_dates:
        months = (max(withdrawal_dates).year - min(injection_dates).year) * 12 \
                 + (max(withdrawal_dates).month - min(injection_dates).month) + 1
    else:
        months = 0
    total_value -= storage_fee_per_month * months

    if total_value < 0:
        print("⚠️ Warning: This contract has a negative expected value")

    return total_value

# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    # Load daily prices (interpolation on historical + forecast)
    nat_gas_daily = pd.read_csv(
        "data/Nat_Gas_Daily.csv",
        parse_dates=['Dates'],
        index_col='Dates'
    )

    contract_value_usd = price_gas_storage_contract(
        price_data=nat_gas_daily,
        injection_dates=["2024-08-31", "2024-09-30"],
        withdrawal_dates=["2024-12-31", "2025-01-31"],
        injection_rate=500_000,
        withdrawal_rate=500_000,
        max_storage=1_000_000,
        storage_fee_per_month=100_000,
        inject_withdraw_fee=0.1,
        transport_fee=50_000
    )

    print("Contract fair value (USD):", round(contract_value_usd, 2))
