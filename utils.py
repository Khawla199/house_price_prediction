
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

scaler = StandardScaler()
label_encoders = {}

def save_scaler_encoders():
    joblib.dump((scaler, label_encoders), "scaler_encoders.joblib")

def load_scaler_encoders():
    global scaler, label_encoders
    scaler, label_encoders = joblib.load("scaler_encoders.joblib")
