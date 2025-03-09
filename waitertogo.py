from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)  # Allow frontend access


file_path = "bills_sample.csv"  
df = pd.read_csv(file_path)

def categorize_bill(size):
    return "Small" if size < 20 else "Medium" if size < 50 else "Large"

df["bill_size_category"] = df["bill_total_billed"].apply(categorize_bill)
df["business_date"] = pd.to_datetime(df["business_date"])
df["tip_percentage"] = (df["payment_total_tip"] / df["bill_total_billed"]) * 100


waiter_analysis = df.groupby("waiter_uuid").agg(
    total_revenue=("bill_total_billed", "sum"),
    total_tips=("payment_total_tip", "sum"),
    avg_order_duration=("order_duration_seconds", "mean"),
    bill_count=("bill_total_billed", "count")
).reset_index()


waiter_features = waiter_analysis[["total_revenue", "total_tips", "avg_order_duration", "bill_count"]].fillna(0)
kmeans = KMeans(n_clusters=3, random_state=42)
waiter_analysis["performance_cluster"] = kmeans.fit_predict(waiter_features)

scaler = MinMaxScaler()
tip_data = df[["business_date", "tip_percentage"]].sort_values("business_date").set_index("business_date")
tip_data_scaled = scaler.fit_transform(tip_data)

sequence_length = 10
X, y = [], []
for i in range(len(tip_data_scaled) - sequence_length):
    X.append(tip_data_scaled[i:i+sequence_length])
    y.append(tip_data_scaled[i+sequence_length])

X, y = np.array(X), np.array(y)

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
    LSTM(50, return_sequences=False),
    Dense(25, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, y, epochs=10, batch_size=16)

future_predictions = model.predict(X[-5:])
future_predictions = scaler.inverse_transform(future_predictions)

@app.route('/api/waiters', methods=['GET'])
def get_waiters():
    return jsonify({
        "waiters": waiter_analysis.to_dict(orient="records"),
        "future_tips": future_predictions.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)

