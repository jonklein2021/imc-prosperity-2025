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
import seaborn as sns

# Read the data
dfs = [
    pd.read_csv(f'data/prices_round_3_day_0.csv', sep=';'),
    pd.read_csv(f'data/prices_round_3_day_1.csv', sep=';'),
    pd.read_csv(f'data/prices_round_3_day_2.csv', sep=';')
]

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

# compute correlations between midprices of each product across all days
pivot = df.pivot_table(index='timestamp', columns='product', values='mid_price')
pivot = pivot.ffill()
correlations = pivot.corr()

# get products in their pivot table order
products = correlations.columns.tolist()

# display results as heatmap
plt.figure(figsize=(10, 8))
plt.xticks(range(len(products)), products, rotation=45)
plt.yticks(range(len(products)), products)
sns.heatmap(correlations, annot=True, cmap='coolwarm', xticklabels=True, yticklabels=True)
plt.title('Correlation between midprices of products')
plt.tight_layout()
plt.show()

