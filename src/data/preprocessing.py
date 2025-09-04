
import numpy as np
import pandas as pd
import os
import re
import string
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def remove_outliers_iqr(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.10)
        Q3 = df[col].quantile(0.90)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df
# Define the preprocessing function
def preprocess(df_main):
    """Apply preprocessing transformations to a comment."""
    try:
        df_main = df_main.drop(columns=["Model"])
        df_main["Engine"] = df_main["Engine"].str.extract(r'(\d+)').astype(float)
        df_main["Power_bhp"] = df_main["Max Power"].str.extract(r'(\d+\.?\d*)').astype(float)
        df_main["Power_rpm"] = df_main["Max Power"].str.extract(r'@ *(\d+)').astype(float)
        df_main.drop(columns=["Max Power"], inplace=True)
        df_main["Power_bhp"].fillna(df_main["Power_bhp"].median(), inplace=True)
        df_main["Power_rpm"].fillna(df_main["Power_rpm"].median(), inplace=True)
        df_main["Torque_Nm"] = df_main["Max Torque"].str.extract(r'(\d+\.?\d*)').astype(float)
        df_main["Torque_rpm"] = df_main["Max Torque"].str.extract(r'@ *(\d+)').astype(float)
        df_main.drop(columns=["Max Torque"], inplace=True)
        df_main["Torque_Nm"].fillna(df_main["Torque_Nm"].median(), inplace=True)
        df_main["Torque_rpm"].fillna(df_main["Torque_rpm"].median(), inplace=True)
        categorical_cols = [
            "Make", "Fuel Type", "Transmission", "Location",
            "Color", "Owner", "Seller Type", "Drivetrain"
        ]

        # Numerical columns
        numeric_cols = [
            "Year", "Kilometer", "Engine", "Length", "Width", "Height",
            "Seating Capacity", "Fuel Tank Capacity",
            "Power_bhp", "Power_rpm", "Torque_Nm", "Torque_rpm"
        ]
        df_main = remove_outliers_iqr(df_main, numeric_cols)
        scaler = StandardScaler()
        df_main[numeric_cols] = scaler.fit_transform(df_main[numeric_cols])
        for col in categorical_cols:
            df_main[col] = LabelEncoder().fit_transform(df_main[col])
        return df_main
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        return df_main



def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the processed train and test datasets."""
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        logger.debug(f"Creating directory {interim_data_path}")
        
        os.makedirs(interim_data_path, exist_ok=True)  # Ensure the directory is created
        logger.debug(f"Directory {interim_data_path} created or already exists")

        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)
        
        logger.debug(f"Processed data saved to {interim_data_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise

def main():
    try:
        logger.debug("Starting data preprocessing...")
        
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded successfully')

        train_processed_data=preprocess(train_data)
        test_processed_data=preprocess(test_data)
        # Save the processed data
        save_data(train_processed_data, test_processed_data, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
