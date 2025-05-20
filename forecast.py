import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("wheat_prices.csv")
df.columns = ['Year', 'Price']
df['Price'] = df['Price'].replace('[\$,]', '', regex=True).astype(float)
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df.set_index('Year', inplace=True)

plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Price'], marker='o', linestyle='solid', color='green')

plt.title("Average Yearly Wheat Prices (1960–2022)", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Price (USD per Bushel)", fontsize=12)
plt.grid(True)
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig("wheat_plot.png")
plt.show()
