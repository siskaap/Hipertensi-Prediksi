import joblib
import numpy as np
import pandas as pd

encoder_cp = joblib.load("model/encoder_cp.joblib")
encoder_exang = joblib.load("model/encoder_exang.joblib")
encoder_fbs = joblib.load("model/encoder_fbs.joblib")
encoder_restecg = joblib.load("model/encoder_restecg.joblib")
encoder_sex = joblib.load("model/encoder_sex.joblib")
encoder_slope = joblib.load("model/encoder_slope.joblib")
encoder_thal = joblib.load("model/encoder_thal.joblib")

scaler_age = joblib.load("model/scaler_age.joblib")
scaler_ca = joblib.load("model/scaler_ca.joblib")
scaler_chol = joblib.load("model/scaler_chol.joblib")
scaler_oldpeak = joblib.load("model/scaler_oldpeak.joblib")
scaler_thalach = joblib.load("model/scaler_thalach.joblib")
scaler_trestbps = joblib.load("model/scaler_trestbps.joblib")

pca_1 = joblib.load("model/pca_1.joblib")

pca_numerical_columns_1 = [
    'thalach',
    'oldpeak'
]

categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

def data_preprocessing(data):

    data = data.copy()
    df = pd.DataFrame()
    
    df["age"] = scaler_age.transform(np.asarray(data["age"]).reshape(-1,1))[0]
    df["sex"] = encoder_sex.transform(data["sex"])
    df["cp"] = encoder_cp.transform(data["cp"])
    df["trestbps"] = scaler_trestbps.transform(np.asarray(data["trestbps"]).reshape(-1,1))[0]
    df["chol"] = scaler_chol.transform(np.asarray(data["chol"]).reshape(-1,1))[0]
    df["fbs"] = encoder_fbs.transform(data["fbs"])
    df["restecg"] = encoder_restecg.transform(data["restecg"])
    df["exang"] = encoder_exang.transform(data["exang"])
    df["slope"] = encoder_slope.transform(data["slope"])
    df["ca"] = scaler_ca.transform(np.asarray(data["ca"]).reshape(-1,1))[0]
    df["thal"] = encoder_thal.transform(data["thal"])

    # PCA 1
    data["thalach"] = scaler_thalach.transform(np.asarray(data["thalach"]).reshape(-1,1))[0]
    data["oldpeak"] = scaler_oldpeak.transform(np.asarray(data["oldpeak"]).reshape(-1,1))[0]
   
    # Fitting PCA
    df["pc_1"] = pca_1.transform(data[pca_numerical_columns_1])
        
    return df