import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

@st.cache_resource
def train_model():
    # Load training data
    df = pd.read_csv("train.csv")
    
    # Separate features and labels (assuming 'class' column is label)
    X = df.drop("class", axis=1)
    y = df["class"]

    # Encode any non-numeric columns automatically
    encoders = {}
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
    
    # If label column is string, encode it too
    if y.dtype == "object":
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)
        encoders["class"] = le_y

    # Train/test split (optional, for training)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X.columns, encoders

# Train or load model
model, feature_columns, encoders = train_model()

st.title("Random Forest Classifier UI")
st.write("Upload a CSV file for classification:")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    # Ensure same feature order
    missing_cols = set(feature_columns) - set(input_df.columns)
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
    else:
        input_df = input_df[feature_columns]
        
        # Apply encoders on uploaded data
        for col in input_df.columns:
            if col in encoders:  # use same encoder as training
                le = encoders[col]
                # If unseen label, replace with -1
                input_df[col] = input_df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        
        predictions = model.predict(input_df)
        
        # Decode predictions if target was encoded
        if "class" in encoders:
            predictions = encoders["class"].inverse_transform(predictions)
        
        st.write("Predictions:")
        st.dataframe(pd.DataFrame({"Prediction": predictions}))
