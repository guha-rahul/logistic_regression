import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# ——— Load artifacts ———
model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')


def load_data():
    df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
    df = df.drop(columns=['id'], errors='ignore')
    df = df.copy()
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    return df

df = load_data()

st.title('Stroke Prediction App')
st.sidebar.title('Navigation')
section = st.sidebar.radio('Go to', ['Home', 'EDA', 'Prediction'])

if section == 'Home':
    st.markdown("""
    ## Welcome  
    This app predicts stroke risk using a logistic regression model.
    - **EDA**: Explore the dataset  
    - **Prediction**: Input your parameters  
    """)

elif section == 'EDA':
    st.header('Exploratory Data Analysis')
    st.write(df.head())
    st.write('### Stroke vs. Gender')
    st.bar_chart(df.groupby('gender')['stroke'].mean())

    st.write('### Age Distribution')
    fig, ax = plt.subplots()
    ax.hist(df['age'], bins=20)
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    
    hist_values = np.histogram(df['age'], bins=20)[0]
    st.bar_chart(hist_values)

elif section == 'Prediction':
    st.header('Make a Prediction')
    st.sidebar.subheader('Input Parameters')

    # Numeric inputs
    age = st.sidebar.slider('Age', 0, 100, 50)
    hypertension = st.sidebar.selectbox('Hypertension (0 = No, 1 = Yes)', [0, 1])
    heart_disease = st.sidebar.selectbox('Heart Disease (0 = No, 1 = Yes)', [0, 1])
    avg_glucose = st.sidebar.slider(
        'Avg. Glucose Level',
        float(df['avg_glucose_level'].min()),
        float(df['avg_glucose_level'].max()),
        float(df['avg_glucose_level'].mean())
    )
    bmi = st.sidebar.slider(
        'BMI',
        float(df['bmi'].min()),
        float(df['bmi'].max()),
        float(df['bmi'].mean())
    )

    # Categorical inputs
    gender = st.sidebar.selectbox('Gender', sorted(df['gender'].unique()))
    ever_married = st.sidebar.selectbox('Ever Married', sorted(df['ever_married'].unique()))
    work_type = st.sidebar.selectbox('Work Type', sorted(df['work_type'].unique()))
    residence = st.sidebar.selectbox('Residence Type', sorted(df['Residence_type'].unique()))
    smoking = st.sidebar.selectbox('Smoking Status', sorted(df['smoking_status'].unique()))

    # Build feature vector
    input_dict = {
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'avg_glucose_level': avg_glucose,
        'bmi': bmi
    }
    input_df = pd.DataFrame([input_dict])

    # One-hot encode with drop_first=True logic
    for col, val in [
        ('gender', gender),
        ('ever_married', ever_married),
        ('work_type', work_type),
        ('Residence_type', residence),
        ('smoking_status', smoking)
    ]:
        cats = sorted(df[col].unique())
        # skip the first category (dropped)
        for cat in cats[1:]:
            col_name = f"{col}_{cat}"
            input_df[col_name] = 1 if val == cat else 0

    # Ensure all model features are present
    for feat in feature_names:
        if feat not in input_df.columns:
            input_df[feat] = 0

    # Reorder and scale
    input_df = input_df[feature_names]
    X_scaled = scaler.transform(input_df.values)

    # Prediction
    prob = model.predict_proba(X_scaled)[0][1]
    pred = model.predict(X_scaled)[0]

    st.markdown(f"**Stroke Probability:** {prob:.2%}")
    st.markdown(f"**Prediction (0=No, 1=Yes):** {pred}")
