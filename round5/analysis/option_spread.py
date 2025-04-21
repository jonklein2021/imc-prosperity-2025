"""
columns
==========================
day
timestamp
product
bid_price_1
bid_volume_1
bid_price_2
bid_volume_2
bid_price_3
bid_volume_3
ask_price_1
ask_volume_1
ask_price_2
ask_volume_2
ask_price_3
ask_volume_3
mid_price
profit_and_loss
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys

"""
columns
==========================
day
timestamp
product
bid_price_1
bid_volume_1
bid_price_2
bid_volume_2
bid_price_3
bid_volume_3
ask_price_1
ask_volume_1
ask_price_2
ask_volume_2
ask_price_3
ask_volume_3
mid_price
profit_and_loss
"""

import pandas as pd
import matplotlib.pyplot as plt
import math
import statistics
import sys

def black_scholes_call(spot, strike, time_to_expiry, volatility):
    d1 = (math.log(spot) - math.log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
    d2 = d1 - volatility * math.sqrt(time_to_expiry)
    call_price = spot * statistics.NormalDist().cdf(d1) - strike * statistics.NormalDist().cdf(d2)
    return call_price

def implied_volatility(call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10):
    low_vol = 0.01
    high_vol = 1.0
    volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
    # binary search ts
    for _ in range(max_iterations):
        estimated_price = black_scholes_call(spot, strike, time_to_expiry, volatility)
        diff = estimated_price - call_price
        if abs(diff) < tolerance:
            break
        elif diff > 0:
            high_vol = volatility
        else:
            low_vol = volatility
        volatility = (low_vol + high_vol) / 2.0
    return volatility

# merge all days together
df = pd.concat([
    pd.read_csv(f'data/prices_round_5_day_2.csv', sep=';'),
    pd.read_csv(f'data/prices_round_5_day_3.csv', sep=';'),
    pd.read_csv(f'data/prices_round_5_day_4.csv', sep=';')
], ignore_index=True)[['day', 'timestamp', 'product', 'mid_price']]
df['timestamp'] += 1000000 * (df['day'] - 2)

options = ['VOLCANIC_ROCK_VOUCHER_9500', 'VOLCANIC_ROCK_VOUCHER_9750', 'VOLCANIC_ROCK_VOUCHER_10000', 'VOLCANIC_ROCK_VOUCHER_10250', 'VOLCANIC_ROCK_VOUCHER_10500']

spot_prices = df[df['product'] == 'VOLCANIC_ROCK'][['timestamp', 'mid_price']].rename(columns={'mid_price': 'spot_price'})

option_dfs = {
    9500: pd.merge(df[df['product'] == 'VOLCANIC_ROCK_VOUCHER_9500'], spot_prices, on='timestamp', how='left'),
    9750: pd.merge(df[df['product'] == 'VOLCANIC_ROCK_VOUCHER_9750'], spot_prices, on='timestamp', how='left'),
    10000: pd.merge(df[df['product'] == 'VOLCANIC_ROCK_VOUCHER_10000'], spot_prices, on='timestamp', how='left'),
    10250: pd.merge(df[df['product'] == 'VOLCANIC_ROCK_VOUCHER_10250'], spot_prices, on='timestamp', how='left'),
    10500: pd.merge(df[df['product'] == 'VOLCANIC_ROCK_VOUCHER_10500'], spot_prices, on='timestamp', how='left')
}

# calculate call price for each option using BS
round_number = 5
for option in options:
    strike = int(option.split('_')[-1])
    option_df = option_dfs[strike]
    option_df['implied_volatility'] = 0.0
    option_df['call_price'] = 0.0
    for i, row in option_df.iterrows():
        spot = row['spot_price']
        tte = ((8 - round_number) / 7) - (row['timestamp'] / 1000000) / 7
        implied_vol = implied_volatility(row['mid_price'], spot, strike, tte)
        call_price = black_scholes_call(spot, strike, tte, implied_vol)
        option_df.at[i, 'call_price'] = call_price
        option_df.at[i, 'spread'] = row['mid_price'] - call_price
    option_dfs[option] = option_df

# plot the spread for each option
fig, axs = plt.subplots(5, 1, figsize=(12, 20))
fig.suptitle('Option Spreads')
for i, option in enumerate(options):
    option_df = option_dfs[option]
    
    # calculate mean and std for spread
    mean_spread = option_df['spread'].mean()
    std_spread = option_df['spread'].std()
    print(f'{option} - Mean Spread: {mean_spread}, Std Spread: {std_spread}')
    
    axs[i].plot(option_df['timestamp'], option_df['spread'], label='Spread')
    axs[i].set_title(f'{option} Spread')
    axs[i].set_xlabel('Timestamp')
    axs[i].set_ylabel('Spread')
    axs[i].legend()
    axs[i].grid()

plt.show()
