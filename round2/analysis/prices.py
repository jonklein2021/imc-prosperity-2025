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

# Read the data
dfs = [
    pd.read_csv(f'data/prices_round_1_day_-2.csv', sep=';'),
    pd.read_csv(f'data/prices_round_1_day_-1.csv', sep=';'),
    pd.read_csv(f'data/prices_round_1_day_0.csv', sep=';')
]

products = ['RAINFOREST_RESIN', 'KELP', 'SQUID_INK', 'DJEMBES', 'CROISSANTS', 'JAMS', 'PICNIC_BASKET1', 'PICNIC_BASKET2']

# Only consider product name if provided
if len(sys.argv) > 1 and sys.argv[1] in products:
    for i in range(len(dfs)):
        dfs[i] = dfs[i][dfs[i]['product'] == sys.argv[1]]

# Only consider day if provided
if len(sys.argv) > 2 and sys.argv[2] in ['-1', '0', '1']:
    df_idx = 2 + int(sys.argv[2])
    
    # Plot mid price against timestamp
    plt.figure(figsize=(10, 5))
    plt.title(f'{sys.argv[1]} MIDPRICE - Day {sys.argv[2]}')
    plt.plot(dfs[df_idx]['timestamp'], dfs[df_idx]['mid_price'], label='Mid Price')
    plt.xlabel('Timestamp')
    plt.ylabel('Mid Price')
    plt.grid()
else:
    # Plot mid price against timestamp
    # Create subplots for each day
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle('MIDPRICE', fontsize=16)
    axs[0].set_title('Day -2')
    axs[1].set_title('Day -1')
    axs[2].set_title('Day 0')
    for i in range(len(dfs)):
        # Plot mid price against timestamp
        axs[i].plot(dfs[i]['timestamp'], dfs[i]['mid_price'], label='Mid Price')
        axs[i].set_ylabel('Mid Price')
        axs[i].legend()
        axs[i].grid()

# Show the plot
plt.tight_layout()
plt.show()
