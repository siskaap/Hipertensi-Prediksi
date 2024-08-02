import streamlit as st
import pandas as pd

# Data Preprocessing
import data_preprocessing as process

# Prediction library
import prediction

st.header('Prediksi Hipertensi (Prototype)')

data = pd.DataFrame() 

col1, col2, col3, col4= st.columns(4) 
 
with col1:
    age = float(st.number_input(label='Umur')) 
    data["age"] = [age]

with col2:
    sex = st.selectbox(label='Sex', options=process.encoder_sex.classes_, placeholder = 'Choose an option', index=None)
    data["sex"] = [sex]
 
with col3:
    cp = st.selectbox(label='Chest Pain', options=process.encoder_cp.classes_, placeholder = 'Choose an option', index=None)
    data["cp"] = [cp]
 
with col4:
    trestbps = float(st.number_input(label='Trest BPS'))
    data["trestbps"] = trestbps

col1, col2, col3, col4= st.columns(4)

with col1:
    chol = float(st.number_input(label='Cholesterol'))
    data["chol"] = [chol]

with col2:
    fbs = st.selectbox(label='Fasting Blood Sugar', options=process.encoder_fbs.classes_, placeholder = 'Choose an option', index=None)
    data["fbs"] = [fbs]

with col3:
    restecg = st.selectbox(label='Resting ECG', options=process.encoder_restecg.classes_, placeholder = 'Choose an option', index=None)
    data["restecg"] = [restecg]
 
with col4:
    thalach = float(st.number_input(label='Thalasemia'))
    data["thalach"] = thalach

col1, col2, col3, col4, col5= st.columns(5)

with col1:
    exang = st.selectbox(label='Exang', options=process.encoder_exang.classes_, placeholder = 'Choose an option', index=None)
    data["exang"] = [exang]

with col2:
    oldpeak = float(st.number_input(label='Old Peak'))
    data["oldpeak"] = oldpeak

with col3:
    slope = st.selectbox(label='Slope', options=process.encoder_slope.classes_, placeholder = 'Choose an option', index=None)
    data["slope"] = slope
 
with col4:
    ca = float(st.number_input(label='Colored by Flourosopy'))
    data["ca"] = ca

with col5:
    thal = st.selectbox(label='Thal', options=process.encoder_thal.classes_, placeholder = 'Choose an option', index=None)
    data["thal"] = thal

with st.expander("Data Masukan"):
    st.dataframe(data=data, width=None, height=10)

if st.button('Predict'):
    proced_data = process.data_preprocessing(data=data)
    with st.expander("Data Hasil Preprocessing"):
        st.dataframe(data=proced_data, width=800, height=10)
    st.title(f"Terprediksi: {prediction.predict(proced_data)} Hipertensi")