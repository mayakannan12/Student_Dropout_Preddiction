from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
CORS(app)

# ===== Load dataset =====
df = pd.read_csv("student_dropout_dataset_v3.csv")

# ===== Handle missing =====
df.fillna(0, inplace=True)

# ===== Convert categorical to numbers =====
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# ===== Target =====
y = df['Dropout']   # (check column name properly!)
X = df.drop(columns=['Dropout'])

# ===== Scaling =====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===== Train =====
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# ===== API =====
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Example input (change based on your dataset columns!)
    sample = pd.DataFrame([data])

    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)

    result = "Dropout" if prediction[0] == 1 else "Not Dropout"

    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)