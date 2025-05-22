from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("wheat_prices.csv")
df.columns = ['Month', 'Price']

start_year = 2015
num_months = len(df)
years = [start_year + (i//12) for i in range(num_months)]

df['Date'] = df['Month'] + ' ' + pd.Series(years).astype(str)
df['Date'] = pd.to_datetime(df['Date'], format = '%B  %Y')
df.set_index('Date', inplace=True)

df = df.copy()
df['Month_Num'] = (df.index.year - 2015) * 12 + (df.index.month - 1)

X = df['Month_Num'].values.reshape(-1, 1)
y = df['Price'].values
model = LinearRegression()
model.fit(X, y)

df['Trendline'] = model.predict(X)

future_months = np.arange(df['Month_Num'].max() + 1, df['Month_Num'].max() + 13).reshape(-1, 1)
future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
future_prices = model.predict(future_months)

future_df = pd.DataFrame({'Price': future_prices, 'Trendline': future_prices}, index=future_dates)

plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Price'], label="Actual Price", color='green')
plt.plot(df.index, df['Trendline'], label="Trendline (Linear Regression)", linestyle='--', color='blue')
plt.plot(future_df.index, future_df['Price'], label="Forecast (Next 12 Months)", linestyle='--', color='red')

df['Moving_Avg_12'] = df['Price'].rolling(window=12).mean()
plt.plot(df.index, df['Moving_Avg_12'], label = '12-Month Moving Avg', color = 'orange', linestyle = '--')

plt.title("Monthly Wheat Prices with Trendline and Forecast")
plt.xlabel("Date")
plt.ylabel("Price (USD per Bushel)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("wheat_forecast_monthly.png")
plt.show()
