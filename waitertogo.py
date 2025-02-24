import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load the CSV file
file_path = "bills_sample.csv"  # Update the path if needed
df = pd.read_csv(file_path)

# Define bill size categories
def categorize_bill(size):
    if size < 20:
        return "Small"
    elif size < 50:
        return "Medium"
    else:
        return "Large"

df["bill_size_category"] = df["bill_total_billed"].apply(categorize_bill)

# Compute waiter performance metrics
waiter_analysis = df.groupby("waiter_uuid").agg(
    total_revenue=("bill_total_billed", "sum"),
    total_tips=("payment_total_tip", "sum"),
    avg_tip_to_bill_ratio=("payment_total_tip", lambda x: (x.sum() / df["bill_total_billed"].sum()) * 100),
    avg_order_duration=("order_duration_seconds", "mean"),
    bill_count=("bill_total_billed", "count")
).reset_index()

# Merge bill size distribution
bill_size_distribution = df.groupby(["waiter_uuid", "bill_size_category"]).size().unstack(fill_value=0)
waiter_analysis = waiter_analysis.merge(bill_size_distribution, on="waiter_uuid", how="left")

# Convert business_date to datetime for seasonal analysis
df["business_date"] = pd.to_datetime(df["business_date"])

# Identify the fastest and slowest waiters
fastest_waiter = df.loc[df["order_duration_seconds"].idxmin(), ["waiter_uuid", "order_duration_seconds"]]
slowest_waiter = df.loc[df["order_duration_seconds"].idxmax(), ["waiter_uuid", "order_duration_seconds"]]

# Categorize waiters as high-tip or low-tip based on a 15% threshold
df["tip_percentage"] = (df["payment_total_tip"] / df["bill_total_billed"]) * 100
df["tip_category"] = df["tip_percentage"].apply(lambda x: "High-Tip" if x >= 15 else "Low-Tip")

# Analyze seasonal tipping trends (weekday vs. weekend)
df["day_of_week"] = df["business_date"].dt.day_name()
df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"])

# Compute average tips for weekdays vs. weekends
seasonal_tipping = df.groupby("is_weekend")["tip_percentage"].mean().reset_index()
seasonal_tipping["is_weekend"] = seasonal_tipping["is_weekend"].map({True: "Weekend", False: "Weekday"})

# Machine Learning - Cluster waiters based on performance
waiter_features = waiter_analysis[["total_revenue", "total_tips", "avg_order_duration", "bill_count"]].fillna(0)
kmeans = KMeans(n_clusters=3, random_state=42)
waiter_analysis["performance_cluster"] = kmeans.fit_predict(waiter_features)

# LSTM Model for Tip Prediction
scaler = MinMaxScaler()
tip_data = df[["business_date", "tip_percentage"]].sort_values("business_date").set_index("business_date")
tip_data_scaled = scaler.fit_transform(tip_data)

# Prepare sequences for LSTM
sequence_length = 10
X, y = [], []
for i in range(len(tip_data_scaled) - sequence_length):
    X.append(tip_data_scaled[i:i+sequence_length])
    y.append(tip_data_scaled[i+sequence_length])
X, y = np.array(X), np.array(y)

# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
    LSTM(50, return_sequences=False),
    Dense(25, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=10, batch_size=16)

# Predict future tip trends
future_predictions = model.predict(X[-5:])
future_predictions = scaler.inverse_transform(future_predictions)

# Print results
print("Waiter Performance Analysis:")
print(waiter_analysis)
print("\nFastest Waiter:")
print(fastest_waiter)
print("\nSlowest Waiter:")
print(slowest_waiter)
print("\nTip Category by Waiter:")
print(df[["waiter_uuid", "tip_category"].drop_duplicates()])
print("\nSeasonal Tipping Trends:")
print(seasonal_tipping)
print("\nWaiter Clusters:")
print(waiter_analysis[["waiter_uuid", "performance_cluster"]])
print("\nFuture Tip Predictions:")
print(future_predictions)


