import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Titanic Survival Analysis")
st.markdown("Exploratory Data Analysis and ML Prediction")

# Load data
@st.cache_data
def load_data():
    df = sns.load_dataset('titanic')
    df['age'].fillna(df['age'].median(), inplace=True)
    df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
    df.drop(columns=['deck', 'embark_town', 'alive', 'who', 'adult_male', 'class'],
            inplace=True, errors='ignore')
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    return df.dropna()

df = load_data()

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", ["Dataset Overview", "EDA Charts", "Predict Survival"])

if page == "Dataset Overview":
    st.subheader("Raw Data Sample")
    st.dataframe(df.head(10))
    st.subheader("Basic Statistics")
    st.write(df.describe())

elif page == "EDA Charts":
    st.subheader("Survival by Gender")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='sex', hue='survived', palette='Set2', ax=ax)
    ax.set_xticklabels(['Male', 'Female'])
    st.pyplot(fig)

    st.subheader("Age Distribution by Survival")
    fig2, ax2 = plt.subplots()
    sns.histplot(data=df, x='age', hue='survived', bins=25, kde=True, ax=ax2)
    st.pyplot(fig2)

elif page == "Predict Survival":
    st.subheader("Predict Your Survival Chance")

    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.radio("Gender", ["Male", "Female"])
    age = st.slider("Age", 1, 80, 25)
    fare = st.slider("Fare Paid", 5, 500, 50)
    sibsp = st.number_input("Siblings/Spouse aboard", 0, 8, 0)
    parch = st.number_input("Parents/Children aboard", 0, 6, 0)

    # Train model
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    X = df[features]
    y = df['survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    st.info(f"Model Accuracy: {acc:.2%}")

    if st.button("Predict"):
        sex_val = 0 if sex == "Male" else 1
        input_data = pd.DataFrame([[pclass, sex_val, age, sibsp, parch, fare, 0]],
                                   columns=features)
        result = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if result == 1:
            st.success(f"Survived! (Probability: {prob:.1%})")
        else:
            st.error(f"Did not survive. (Survival probability: {prob:.1%})")
