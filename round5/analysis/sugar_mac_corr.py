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

# merge all days together
prices_df = pd.concat([
    pd.read_csv(f'data/prices_round_5_day_2.csv', sep=';'),
    pd.read_csv(f'data/prices_round_5_day_3.csv', sep=';'),
    pd.read_csv(f'data/prices_round_5_day_4.csv', sep=';')
], ignore_index=True)[['day', 'timestamp', 'product', 'mid_price']]
prices_df['timestamp'] += 1000000 * (prices_df['day'] - 2)


