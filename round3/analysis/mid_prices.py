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
import numpy as np

# Read the data
dfs = [
    pd.read_csv(f'data/prices_round_3_day_0.csv', sep=';'),
    pd.read_csv(f'data/prices_round_3_day_1.csv', sep=';'),
    pd.read_csv(f'data/prices_round_3_day_2.csv', sep=';')
]

products = [
    'RAINFOREST_RESIN', 'KELP', 'SQUID_INK',
    'DJEMBES', 'CROISSANTS', 'JAMS', 'PICNIC_BASKET1', 'PICNIC_BASKET2',
    'VOLCANIC_ROCK', 'VOLCANIC_ROCK_VOUCHER_9500', 'VOLCANIC_ROCK_VOUCHER_9750', 'VOLCANIC_ROCK_VOUCHER_10000', 'VOLCANIC_ROCK_VOUCHER_10250', 'VOLCANIC_ROCK_VOUCHER_10500',
]

# calculate statistics about midprices of each product, for each day
for day in range(0, 3, 1):
    print(f'== DAY {day} ==')
    for p in products:
        # calculate statistics about mid prices of a product
        df = dfs[day]
        
        midprices = df[df['product'] == p]['mid_price'].dropna()
        if len(midprices) > 0:
            mean = round(np.mean(midprices), 4)
            median = round(np.median(midprices), 4)
            stddev = round(np.std(midprices), 4)
            print(f'{p} Mid Price: mean={mean}, median={median}, stddev={stddev}')
        else:
            print(f'No data for {p}')
    print("")

print('== ALL DAYS ==')

# calculate same statistics for all days
for p in products:
    # calculate statistics about mid prices of a product
    midprices = pd.concat([df[df['product'] == p]['mid_price'].dropna() for df in dfs], ignore_index=True)
    if len(midprices) > 0:
        mean = round(np.mean(midprices), 4)
        median = round(np.median(midprices), 4)
        stddev = round(np.std(midprices), 4)
        print(f'{p} Mid Price: mean={mean}, median={median}, stddev={stddev}')
    else:
        print(f'No data for {p}')
