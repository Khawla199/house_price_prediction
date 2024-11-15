import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectFromModel
from app.utils import label_encoders, scaler
import xgboost as xgb


# Function to fill missing values in the dataset
def fill_missing_values(data):
    if data is None or data.empty:
        raise ValueError("Input data is None or empty. Please provide valid data.")
    
    data = data.copy()  # Avoid modifying the original DataFrame
    # Fill missing values for continuous and categorical columns
    data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
    data['MasVnrArea'] = data['MasVnrArea'].fillna(0)
    data[['BsmtQual', 'BsmtCond']] = data[['BsmtQual', 'BsmtCond']].fillna('NA')
    data['BsmtExposure'] = data['BsmtExposure'].fillna('No')
    data['Electrical'].fillna(data['Electrical'].mode()[0], inplace=True)
    data['GarageYrBlt'].fillna(0, inplace=True)
    data['GarageFinish'].fillna('NA', inplace=True)
    data['FireplaceQu'].fillna('NA', inplace=True)
    return data


# Preprocess function for both training and testing datasets
def preprocess_data(train_data, test_data=None, is_training=True, feature_names=None):
    # Define continuous and categorical features
    continuous_features = [
        'LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt', 'YearRemodAdd',
        'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
        'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
        'GarageCars', 'GarageArea', 'OpenPorchSF'
    ]
    categorical_features = [
        'MSZoning', 'LotShape', 'LandSlope', 'ExterQual', 'ExterCond',
        'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'HeatingQC',
        'CentralAir', 'Electrical', 'KitchenQual', 'FireplaceQu', 'GarageFinish',
        'SaleType', 'SaleCondition'
    ]

    # Preprocessing for the training dataset
    if is_training:
        train_data = fill_missing_values(train_data)
        X = train_data[continuous_features + categorical_features].copy()
        y = train_data['SalePrice']

        # Standardize continuous features
        scaler.fit(X[continuous_features])
        X[continuous_features] = scaler.transform(X[continuous_features])

        # Encode categorical features
        for col in categorical_features:
            label_encoders[col] = LabelEncoder().fit(X[col].astype(str))
            X[col] = label_encoders[col].transform(X[col].astype(str))

        # Feature selection using XGBoost
        selector = SelectFromModel(xgb.XGBRegressor(n_estimators=100), threshold='median').fit(X, y)
        selected_features = X.columns[selector.get_support()]

        return selector.transform(X), y, None, selected_features

    # Preprocessing for the testing dataset
    else:
        if feature_names is None:
            raise ValueError("Feature names are required for testing data preprocessing.")
        
        test_data = fill_missing_values(test_data)
        X_test = test_data[feature_names].copy()  # Ensure we use the same features as training

        # Standardize continuous features
        X_test[continuous_features] = scaler.transform(X_test[continuous_features])

        # Encode categorical features
        for col in categorical_features:
            if col not in label_encoders:
                raise ValueError(f"Label encoder for column '{col}' is not found. Ensure training has been completed.")
            X_test[col] = label_encoders[col].transform(X_test[col].astype(str))

        return X_test
