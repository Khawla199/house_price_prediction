
import xgboost as xgb
import joblib
from sklearn.metrics import root_mean_squared_error
from app.preprocess import preprocess_data
from app.utils import load_scaler_encoders, save_scaler_encoders

def train_model(train_data, test_data):
    X_train, y_train, X_test, selected_feature_names = preprocess_data(train_data, test_data)
    
    # Train the model
    xgb_model = xgb.XGBRegressor(
        colsample_bytree=0.6, learning_rate=0.05, max_depth=5,
        n_estimators=300, subsample=0.8, random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    # Save model, scaler, and encoders
    joblib.dump((xgb_model, selected_feature_names), "xgb_model.joblib")
    save_scaler_encoders()
    
    # Evaluate the model
    y_pred = xgb_model.predict(X_train)
    rmse = root_mean_squared_error(y_train, y_pred, squared=False)
    return rmse

def make_predictions(test_data):
    xgb_model, selected_feature_names = joblib.load("xgb_model.joblib")
    X_test = preprocess_data(test_data, is_training=False, feature_names=selected_feature_names)
    
    # Predict
    predictions = xgb_model.predict(X_test)
    return predictions.tolist()
