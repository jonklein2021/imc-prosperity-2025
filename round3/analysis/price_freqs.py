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
import sys

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

# Only consider product name if provided
if len(sys.argv) > 1 and sys.argv[1] in products:
    for i in range(len(dfs)):
        dfs[i] = dfs[i][dfs[i]['product'] == sys.argv[1]]

# Only consider day if provided
if len(sys.argv) > 2 and sys.argv[2] in {'-1', '0', '1', 'all', 'ALL'}:
    plt.figure(figsize=(10, 5))
    
    # If 'all' is provided, plot all days and merge dataframes
    if sys.argv[2].lower() == 'all':
        dfs = pd.concat(dfs, ignore_index=True)
        df_idx = 0
        
        # Plot histogram of mid price
        plt.title(f'{sys.argv[1]} MIDPRICE - All Days')
        plt.hist(dfs['mid_price'], bins=250, label='Mid Price')
        
        
        # Plot normal distribution on same plot
        mu = dfs['mid_price'].mean()
        sigma = dfs['mid_price'].std()
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        y = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2) * len(dfs)
        plt.plot(x, y, 'r--', label='Normal Distribution')
        
        plt.xlabel('Mid Price')
    else:
        df_idx = 2 + int(sys.argv[2])
        
        # Plot histogram of mid price
        plt.title(f'{sys.argv[1]} MIDPRICE - Day {sys.argv[2]}')
        plt.hist(dfs[df_idx]['mid_price'], bins=250, label='Mid Price')
        
        # Plot normal distribution on same plot
        mu = dfs[df_idx]['mid_price'].mean()
        sigma = dfs[df_idx]['mid_price'].std()
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        y = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2) * len(dfs[df_idx]) # scale by number of samples
        plt.plot(x, y, 'r--', label='Normal Distribution')
        
        plt.xlabel('Mid Price')
else:
    # Plot mid price against timestamp
    # Create subplots for each day
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    # fig.suptitle('MIDPRICE', fontsize=16)
    axs[0].set_title('Day -2')
    axs[1].set_title('Day -1')
    axs[2].set_title('Day 0')
    for i in range(len(dfs)):
        # Plot histogram of mid price
        axs[i].hist(dfs[i]['mid_price'], bins=250, label='Mid Price')
        
        # Plot normal distribution on same plot
        mu = dfs[i]['mid_price'].mean()
        sigma = dfs[i]['mid_price'].std()
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        y = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2) * len(dfs[i])
        axs[i].plot(x, y, 'r--', label='Normal Distribution')
        
        axs[i].grid()

# Show the plot
plt.tight_layout()
plt.show()
