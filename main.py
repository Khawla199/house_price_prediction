from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
from app.model import make_predictions

app = FastAPI()

# Define the file paths to the training and testing datasets
TRAIN_FILE_PATH = r'C:\Users\k.alquraan.ext\Desktop\my_fastapi_project\train.csv'
TEST_FILE_PATH = r'C:\Users\k.alquraan.ext\Desktop\my_fastapi_project\test.csv'

# Define the input data model
class PredictionInput(BaseModel):
    data: list  # Expecting a list of dictionaries

@app.post("/predict")
async def predict(input_data: PredictionInput):
    # Debug: Print the received input data
    print(f"Received input data: {input_data}")

    if not input_data.data or len(input_data.data) == 0:
        raise HTTPException(status_code=400, detail="Input data is empty or invalid.")

    # Convert the input data (which is a list of dictionaries) into a DataFrame
    try:
        input_df = pd.DataFrame(input_data.data)
        print(f"Converted input data to DataFrame: {input_df}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error converting input data to DataFrame: {str(e)}")

    if input_df.empty:
        raise HTTPException(status_code=400, detail="Input DataFrame is empty.")

    # Ensure that the required features are present in the DataFrame
    required_features = [
        'LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt', 'YearRemodAdd',
        'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
        'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
        'GarageCars', 'GarageArea', 'OpenPorchSF'
    ]

    missing_features = [feature for feature in required_features if feature not in input_df.columns]

    if missing_features:
        raise HTTPException(status_code=400, detail=f"Missing required features: {', '.join(missing_features)}")

    # Make predictions using the trained model (assuming make_predictions is implemented)
    try:
        predictions = make_predictions(input_df)
        print(f"Predictions: {predictions}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return {"predictions": predictions}