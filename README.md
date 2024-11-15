Here's a detailed README file for the entire house price prediction task:


### README.md

# House Price Prediction Project

This repository provides a comprehensive solution for predicting house prices using a machine learning pipeline built with XGBoost. It includes data preprocessing, feature selection, model training, and deployment using FastAPI.

## Overview

The goal of this project is to predict house prices based on features from a dataset. The project involves:
1. **Data preprocessing**: Handling missing values and encoding categorical variables.
2. **Feature selection**: Identifying the most relevant features using an XGBoost-based model.
3. **Model training and evaluation**: Using XGBoost regression to predict house prices.
4. **Deployment**: Building a FastAPI-based application to train the model and make predictions.


## Project Features

### 1. Data Preprocessing
- Filling missing values for both continuous and categorical variables.
- Scaling continuous features using `StandardScaler`.
- Encoding categorical variables with `LabelEncoder`.

### 2. Model Training
- Trains an XGBoost regressor with optimal hyperparameters.
- Automatically selects the most important features based on importance thresholds.
- Saves the trained model for future use.

### 3. Prediction
- Generates predictions for the test dataset.
- Outputs predictions in CSV format.

### 4. Deployment
- FastAPI is used to expose two endpoints:
  - **/train**: Train the model using uploaded training and testing datasets.
  - **/predict**: Generate predictions using the uploaded test dataset.


## Requirements

- Python 3.x
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```


## Endpoints

### Train the Model

- **Endpoint**: `POST /train`
- **Description**: Trains the XGBoost model with the provided datasets.
- **Inputs**: 
  - `train_file`: Training dataset (CSV file).
  - `test_file`: Testing dataset (CSV file).
- **Response**:
  - Root Mean Squared Error (RMSE) of the training process.
  - List of features selected during training.

- **Example cURL**:
  ```bash
  curl -X POST "http://127.0.0.1:8000/train" \
       -F "train_file=@path/to/train.csv" \
       -F "test_file=@path/to/test.csv"
  ```


### Make Predictions

- **Endpoint**: `POST /predict`
- **Description**: Generates predictions for the provided test dataset.
- **Inputs**:
  - `test_file`: Test dataset (CSV file).
- **Response**:
  - CSV file containing predictions.

- **Example cURL**:
  ```bash
  curl -X POST "http://127.0.0.1:8000/predict" \
       -F "test_file=@path/to/test.csv" \
       --output predictions.csv
  ```


## Folder Structure

```
house_price_prediction/
my_fastapi_project/
│

├── __init__.py  (optional if inside a package)
├── main.py      # Contains FastAPI app
├── model.py     # Contains train_model, make_predictions
├── preprocess.py
└── utils.py
│
├── train.csv
└── test.csv
│
├── requirements.txt
└── ...



## How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Server
Start the FastAPI server locally:
```bash
uvicorn main:app --reload
```

### Step 3: Train the Model
Use the `/train` endpoint to train the model using your datasets.

### Step 4: Make Predictions
Use the `/predict` endpoint to generate predictions for the test dataset.


## Model Details

- **Algorithm**: XGBoost Regression
- **Evaluation Metric**: Root Mean Squared Error (RMSE)
- **Feature Selection**: Based on feature importance with a median threshold.

