import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
df = pd.read_csv("sales_data.csv")

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

print(df.head())
print("Missing values:\n", df.isnull().sum())

df['sales'] = df['sales'].fillna(method='ffill')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek

df['lag_1'] = df['sales'].shift(1)
df['lag_7'] = df['sales'].shift(7)
df['rolling_7'] = df['sales'].rolling(7).mean()

df.dropna(inplace=True)

print(df.head())
train_size = int(len(df) * 0.8)

train = df[:train_size]
test = df[train_size:]

X_train = train.drop(['date', 'sales'], axis=1)
y_train = train['sales']

X_test = test.drop(['date', 'sales'], axis=1)
y_test = test['sales']
model = LinearRegression()
model.fit(X_train, y_train)
test['forecast'] = model.predict(X_test)
mae = mean_absolute_error(y_test, test['forecast'])
rmse = np.sqrt(mean_squared_error(y_test, test['forecast']))

print("MAE:", mae)
print("RMSE:", rmse)
plt.figure(figsize=(12,6))

plt.plot(train['date'], train['sales'], label="Training Sales")
plt.plot(test['date'], test['sales'], label="Actual Sales")
plt.plot(test['date'], test['forecast'], label="Forecasted Sales")

plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Forecasting Model")
plt.legend()
plt.grid(True)
plt.show()
future_days = 15
last_row = df.iloc[-1].copy()

future_sales = []

for i in range(future_days):
    features = last_row.drop(['date', 'sales'])
    pred = model.predict([features])[0]
    
    future_sales.append(pred)
    
    last_row['sales'] = pred
    last_row['lag_7'] = last_row['lag_1']
    last_row['lag_1'] = pred

future_dates = pd.date_range(
    start=df['date'].iloc[-1], periods=future_days+1, freq='D'
)[1:]

future_df = pd.DataFrame({
    'date': future_dates,
    'forecasted_sales': future_sales
})

print(future_df)
plt.figure(figsize=(12,6))

plt.plot(df['date'], df['sales'], label="Historical Sales")
plt.plot(future_df['date'], future_df['forecasted_sales'],
         linestyle='dashed', label="Future Forecast")

plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Future Sales Forecast")
plt.legend()
plt.grid(True)
plt.show()