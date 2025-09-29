import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
df = pd.read_csv('electricity_bill.csv')

# Prepare features and target
X = df[['Units_Consumed']]
Y = df['Electricity_Bill']

# Split data for training/testing (optional)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, Y_train)

# Save model for later use
joblib.dump(model, 'electricity_bill_predictor_model.pkl')
print("Model trained and saved successfully.")
