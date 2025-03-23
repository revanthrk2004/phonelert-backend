import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# STEP 1: Prepare your data
# CSV file with columns: latitude, longitude, alert (1 = alert, 0 = no alert)
data = pd.read_csv("location_data.csv")

X = data[["latitude", "longitude"]]
y = data["alert"]

# STEP 2: Train a model
model = RandomForestClassifier()
model.fit(X, y)

# STEP 3: Save the model
joblib.dump(model, "location_alert_model.pkl")

print("✅ Model trained and saved as location_alert_model.pkl")
