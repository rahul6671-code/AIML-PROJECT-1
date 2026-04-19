import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# ---------------------------

# PAGE CONFIG

# ---------------------------

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

st.title("📊 Employee Attrition Prediction App")

# ---------------------------

# LOAD DATA

# ---------------------------

@st.cache_data
def load_data():
url = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
return pd.read_csv(url)

df = load_data()

st.subheader("📋 Dataset Preview")
st.dataframe(df.head())

# ---------------------------

# PREPROCESSING

# ---------------------------

df_model = df.copy()

df_model['Attrition'] = (df_model['Attrition'] == 'Yes').astype(int)

binary_map = {'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0}
for col in ['Gender', 'OverTime']:
if col in df_model.columns:
df_model[col] = df_model[col].map(binary_map)

df_model = pd.get_dummies(df_model, drop_first=True)

X = df_model.drop('Attrition', axis=1)
y = df_model['Attrition']

# ---------------------------

# TRAIN MODEL

# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_sm, y_train_sm)

# ---------------------------

# USER INPUT

# ---------------------------

st.sidebar.header("🧑 Employee Details")

age = st.sidebar.slider("Age", 18, 60, 30)
income = st.sidebar.number_input("Monthly Income", 1000, 20000, 5000)
distance = st.sidebar.slider("Distance From Home", 1, 30, 10)
job_level = st.sidebar.slider("Job Level", 1, 5, 2)
job_sat = st.sidebar.slider("Job Satisfaction", 1, 4, 2)
overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])

# ---------------------------

# CREATE INPUT DATA

# ---------------------------

input_data = pd.DataFrame({
'Age': [age],
'MonthlyIncome': [income],
'DistanceFromHome': [distance],
'JobLevel': [job_level],
'JobSatisfaction': [job_sat],
'OverTime': [1 if overtime == "Yes" else 0]
})

# Match columns

input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# ---------------------------

# PREDICTION

# ---------------------------

if st.sidebar.button("Predict Attrition"):
prob = model.predict_proba(input_data)[0][1]

```
st.subheader("📊 Prediction Result")

st.write(f"**Attrition Probability:** {prob:.2%}")

if prob > 0.7:
    st.error("🔴 High Risk of Leaving")
elif prob > 0.4:
    st.warning("🟡 Medium Risk")
else:
    st.success("🟢 Low Risk")
```

# ---------------------------

# MODEL PERFORMANCE

# ---------------------------

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("📈 Model Performance")
st.write(f"Accuracy: {acc:.2%}")
