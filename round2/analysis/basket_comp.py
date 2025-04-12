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
import numpy as np
import statistics
import sys

# Read the data
dfs = [
    pd.read_csv(f'data/prices_round_2_day_-1.csv', sep=';'),
    pd.read_csv(f'data/prices_round_2_day_0.csv', sep=';'),
    pd.read_csv(f'data/prices_round_2_day_1.csv', sep=';')
]

products = ['DJEMBES', 'CROISSANTS', 'JAMS', 'PICNIC_BASKET1', 'PICNIC_BASKET2']

# merge dataframes
df = pd.concat(dfs, ignore_index=True)

# day 0: add 1000000 to timestamp
df.loc[df['day'] == 0, 'timestamp'] += 1000000

# day 1: add 2000000 to timestamp
df.loc[df['day'] == 1, 'timestamp'] += 2000000

# drop unnecessary columns
df = df.drop(columns=[
    'day', 'bid_price_1', 'bid_volume_1', 'bid_price_2',
    'bid_volume_2', 'bid_price_3', 'bid_volume_3', 'ask_price_1',
    'ask_volume_1', 'ask_price_2', 'ask_volume_2', 'ask_price_3',
    'ask_volume_3', 'profit_and_loss'
])

# PICNIC_BASKET1 = 6 CROISSANTS, 3 JAMS, 1 DJEMBE
# PICNIC_BASKET2 = 4 CROISSANTS, 2 JAMS

# create synthetic picnic basket rows from existing row
def calculate_basket_prices(row):
    try:
        croissants_price = row[row['product'] == 'CROISSANTS']['mid_price'].values[0]
        jams_price = row[row['product'] == 'JAMS']['mid_price'].values[0]
        djembes_price = row[row['product'] == 'DJEMBES']['mid_price'].values[0]
        return [
            {'timestamp': row['timestamp'].iloc[0], 'product': 'SYNTHETIC_BASKET1', 'mid_price': croissants_price * 6 + jams_price * 3 + djembes_price},
            {'timestamp': row['timestamp'].iloc[0], 'product': 'SYNTHETIC_BASKET2', 'mid_price': croissants_price * 4 + jams_price * 2}
        ]
    except IndexError:
        return []

synthetic_rows = df.groupby('timestamp').apply(calculate_basket_prices).explode().dropna().tolist()
df = pd.concat([df, pd.DataFrame(synthetic_rows)], ignore_index=True)

# plot synthetic baskets and real baskets
fig, axs = plt.subplots(2, 1, figsize=(10, 15))
timeline = sorted(df['timestamp'].unique())

# picnic basket 1 and synthetic basket 1
axs[0].plot(timeline, df[df['product'] == 'SYNTHETIC_BASKET1']['mid_price'], label='Synthetic Basket 1', color='lightcoral')
axs[0].plot(timeline, df[df['product'] == 'PICNIC_BASKET1']['mid_price'], label='Picnic Basket 1', color='red')
axs[0].grid()
axs[0].legend()

# picnic basket 2 and synthetic basket 2
axs[1].plot(timeline, df[df['product'] == 'SYNTHETIC_BASKET2']['mid_price'], label='Synthetic Basket 2', color='skyblue')
axs[1].plot(timeline, df[df['product'] == 'PICNIC_BASKET2']['mid_price'], label='Picnic Basket 2', color='blue')
axs[1].grid()
axs[1].legend()

plt.tight_layout()
plt.show()

