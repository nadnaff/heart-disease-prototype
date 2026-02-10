import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("heart_disease_model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        return model, scaler
    except:
        return None, None

model, scaler = load_model()

# =========================
# SIDEBAR NAVIGATION
# =========================
page = st.sidebar.radio(
    "Menu",
    ["Prediction", "Data", "About"]
)

# =========================
# PREDICTION PAGE
# =========================
if page == "Prediction":

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 20, 100, 50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        sex_val = 1 if sex == "Male" else 0

        trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
        chol = st.slider("Cholesterol", 100, 600, 200)

        fbs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
        fbs_val = 1 if fbs == "Yes" else 0

        thalach = st.slider("Max Heart Rate", 60, 220, 150)

    with col2:
        cp = st.selectbox("Chest Pain Type", [0,1,2,3])
        restecg = st.selectbox("Rest ECG", [0,1,2])
        exang = st.selectbox("Exercise Angina", ["No", "Yes"])
        exang_val = 1 if exang == "Yes" else 0

        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
        slope = st.selectbox("Slope", [0,1,2])
        ca = st.slider("Major Vessels", 0, 4, 0)
        thal = st.selectbox("Thal", [0,1,2,3])

    if st.button("Predict", use_container_width=True):

        if model is None:
            st.error("Model belum tersedia")
            st.stop()

        features = np.array([[
            age, sex_val, cp, trestbps, chol, fbs_val,
            restecg, thalach, exang_val, oldpeak,
            slope, ca, thal
        ]])

        scaled = scaler.transform(features)
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]

        colA, colB = st.columns(2)

        with colA:
            if pred == 0:
                st.success("Low risk")
            else:
                st.error("High risk")

            st.write("Probability:", round(prob*100,2), "%")

        with colB:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob*100,
                gauge={"axis":{"range":[0,100]}}
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        df = pd.DataFrame({
            "Class":["No Disease","Disease"],
            "Probability":[(1-prob)*100, prob*100]
        })

        fig2 = px.bar(df, x="Class", y="Probability")
        st.plotly_chart(fig2, use_container_width=True)

# =========================
# DATA PAGE
# =========================
elif page == "Data":

    st.write("Feature Information")

    info = {
        "age":"Age",
        "sex":"Gender",
        "cp":"Chest pain type",
        "trestbps":"Blood pressure",
        "chol":"Cholesterol",
        "fbs":"Fasting sugar",
        "restecg":"ECG",
        "thalach":"Heart rate",
        "exang":"Angina",
        "oldpeak":"ST depression",
        "slope":"Slope",
        "ca":"Vessels",
        "thal":"Thal"
    }

    st.dataframe(pd.DataFrame(info.items(), columns=["Feature","Description"]))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "85%")
    col2.metric("Precision", "84%")
    col3.metric("Recall", "87%")
    col4.metric("F1", "85%")

    importance = pd.DataFrame({
        "Feature":["ca","cp","thalach","oldpeak","thal","age"],
        "Importance":[0.15,0.14,0.13,0.12,0.11,0.10]
    }).sort_values("Importance")

    fig = px.bar(importance, x="Importance", y="Feature", orientation="h")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# ABOUT
# =========================
else:

    st.write("Heart Disease Prediction System")
    st.write("Machine learning model for prediction based on clinical data.")
    st.write("For educational use only.")
