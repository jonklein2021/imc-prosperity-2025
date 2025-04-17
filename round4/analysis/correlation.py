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

# Read the data, merge dataframes
df = pd.concat([
    pd.read_csv(f'data/prices_round_4_day_1.csv', sep=';'),
    pd.read_csv(f'data/prices_round_4_day_2.csv', sep=';'),
    pd.read_csv(f'data/prices_round_4_day_3.csv', sep=';')
], ignore_index=True)[['day', 'timestamp', 'product', 'mid_price']]

# adjust timestamps
df['timestamp'] += 1000000 * (df['day'] - 1)

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
