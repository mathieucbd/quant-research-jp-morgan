import pandas as pd

def price_gas_storage_contract(
    price_data: pd.DataFrame,      # DataFrame with 'Prices' column and datetime index
    injection_dates: list,         # list of monthly YYYY-MM-DD dates
    withdrawal_dates: list,        # list of monthly YYYY-MM-DD dates
    injection_rate: float,         # MMBtu per injection month
    withdrawal_rate: float,        # MMBtu per withdrawal month
    max_storage: float,            # MMBtu (max capacity)
    storage_fee_per_month: float,  # USD/month
    inject_withdraw_fee: float,    # USD/MMBtu (variable)
    transport_fee: float           # USD per injection/withdrawal event
) -> float:
    """
    Calculate the fair value (USD) of a natural gas storage contract on monthly data.
    
    """

    total_value = 0.0
    stored_volume = 0.0

    # Convert date strings to Timestamps
    injection_dates = [pd.Timestamp(d) for d in injection_dates]
    withdrawal_dates = [pd.Timestamp(d) for d in withdrawal_dates]

    # --- Injection phase ---
    for date in injection_dates:
        date = pd.Timestamp(date)
        if date not in price_data.index:
            raise ValueError(f"Injection date {date} not in price data.")
        
        price = price_data.loc[date, 'Prices']
        volume = min(injection_rate, max_storage - stored_volume)
        stored_volume += volume

        # Calculate costs
        total_value -= price * volume
        total_value -= inject_withdraw_fee * volume - transport_fee

    # --- Withdrawal phase ---
    for date in withdrawal_dates:
        date = pd.Timestamp(date)
        if date not in price_data.index:
            raise ValueError(f"Withdrawal date {date} not in price data.")
        
        price = price_data.loc[date, 'Prices']
        volume = min(withdrawal_rate, stored_volume)
        stored_volume -= volume

        # Calculate revenues
        total_value += price * volume
        total_value -= inject_withdraw_fee * volume - transport_fee

    # --- Storage fees ---
    if injection_dates and withdrawal_dates:
        start_date = min(injection_dates)
        end_date = max(withdrawal_dates)
        months = int(end_date - start_date).days // 30 + 1
    else:
        months = 0

    total_value -= storage_fee_per_month * months

    return total_value


    

