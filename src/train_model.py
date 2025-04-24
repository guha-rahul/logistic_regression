import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Create models/ directory
os.makedirs('models', exist_ok=True)

# 1) Load and clean data
df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
df = df.drop(columns=['id'], errors='ignore')
df['bmi'].fillna(df['bmi'].median(), inplace=True)

# 2) Define features
categorical = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
numerical = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']

# 3) One-hot encode
df_enc = pd.get_dummies(df, columns=categorical, drop_first=True)
feature_names = df_enc.drop(columns=['stroke']).columns.tolist()

# 4) Split X/y
X = df_enc.drop(columns=['stroke']).values
y = df_enc['stroke'].values

# 5) Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6) Train/test split and fit
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
model = LogisticRegression()
model.fit(X_train, y_train)

# 7) Save artifacts
joblib.dump(model, 'models/model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(feature_names, 'models/feature_names.pkl')
print("Training complete. Artifacts saved in models/")
