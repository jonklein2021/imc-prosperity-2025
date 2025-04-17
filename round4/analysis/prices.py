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
df = pd.concat([
    pd.read_csv(f'data/prices_round_4_day_1.csv', sep=';'),
    pd.read_csv(f'data/prices_round_4_day_2.csv', sep=';'),
    pd.read_csv(f'data/prices_round_4_day_3.csv', sep=';')
], ignore_index=True)[['day', 'timestamp', 'product', 'mid_price']]
df['timestamp'] += 1000000 * (df['day'] - 1)

products = [
    'RAINFOREST_RESIN', 'KELP', 'SQUID_INK',
    'DJEMBES', 'CROISSANTS', 'JAMS', 'PICNIC_BASKET1', 'PICNIC_BASKET2',
    'VOLCANIC_ROCK', 'VOLCANIC_ROCK_VOUCHER_9500', 'VOLCANIC_ROCK_VOUCHER_9750', 'VOLCANIC_ROCK_VOUCHER_10000', 'VOLCANIC_ROCK_VOUCHER_10250', 'VOLCANIC_ROCK_VOUCHER_10500',
    'MAGNIFICENT_MACARONS'
]

# select product
if len(sys.argv) > 1 and sys.argv[1] in products:
    df = df[df['product'] == sys.argv[1]]
else:
    print("Error: Invalid product name or product not provided")
    exit(1)

# filter by day if provided
if len(sys.argv) > 2 and sys.argv[2] in {'0', '1', '2'}:
    df = df[df['day'] == int(sys.argv[2])]

plt.figure(figsize=(12, 6))
plt.title(f'{sys.argv[1]} Midprice{" - Day " + sys.argv[2] if len(sys.argv) > 2 else ""}')

# plot the mid price
plt.plot(df['timestamp'], df['mid_price'], label='Mid Price')

# plot SMA
plt.plot(df['timestamp'], df['mid_price'].rolling(window=500).mean(), label='SMA (500)')

plt.legend()
plt.xlabel('Timestamp')
plt.ylabel('Mid Price')
plt.grid()
plt.tight_layout()
plt.show()
