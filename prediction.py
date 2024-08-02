import joblib
 
model = joblib.load("model/rf_model.joblib")
result_target = joblib.load("model/encoder_target.joblib")

def predict(data):
    result = model.predict(data)
    final_result = result_target.inverse_transform(result)[0]
    return final_result