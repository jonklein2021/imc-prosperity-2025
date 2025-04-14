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
    pd.read_csv(f'data/prices_round_3_day_0.csv', sep=';'),
    pd.read_csv(f'data/prices_round_3_day_1.csv', sep=';'),
    pd.read_csv(f'data/prices_round_3_day_2.csv', sep=';')
]

# drop unnecessary columns
dfs = [df.drop(columns=[
    'bid_price_1', 'bid_volume_1', 'bid_price_2',
    'bid_volume_2', 'bid_price_3', 'bid_volume_3', 'ask_price_1',
    'ask_volume_1', 'ask_price_2', 'ask_volume_2', 'ask_price_3',
    'ask_volume_3', 'profit_and_loss'
]) for df in dfs]

vouchers = ['VOLCANIC_ROCK_VOUCHER_9500', 'VOLCANIC_ROCK_VOUCHER_9750', 'VOLCANIC_ROCK_VOUCHER_10000', 'VOLCANIC_ROCK_VOUCHER_10250', 'VOLCANIC_ROCK_VOUCHER_10500']

# Only consider day if provided
if len(sys.argv) > 1 and sys.argv[1] in {'0', '1', '2', 'all', 'ALL'}:
    if sys.argv[1].lower() == 'all':
        dfs = pd.concat(dfs, ignore_index=True)
              
        # day 1: add 1000000 to timestamp
        dfs.loc[dfs['day'] == 1, 'timestamp'] += 1000000

        # day 2: add 2000000 to timestamp
        dfs.loc[dfs['day'] == 2, 'timestamp'] += 2000000
        
        # Plot mid price of each voucher against timestamp
        frames = {
            v: dfs[dfs['product'] == v] for v in vouchers
        }
        
        plt.figure(figsize=(12, 6))
        plt.title(f'MIDPRICE - Day {sys.argv[1]}')
        for v, df in frames.items():
            plt.plot(df['timestamp'], df['mid_price'], label=v)
        plt.legend()
        plt.xlabel('Timestamp')
        plt.ylabel('Mid Price')
        plt.grid()
    else:
        df_idx = int(sys.argv[2])
        
        # Plot mid price against timestamp
        plt.figure(figsize=(10, 5))
        plt.title(f'{sys.argv[1]} MIDPRICE - Day {sys.argv[2]}')
        plt.plot(dfs[df_idx]['timestamp'], dfs[df_idx]['mid_price'], label='Mid Price')
        plt.legend()
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
