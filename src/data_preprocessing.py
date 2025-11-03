"""
data_preprocessing.py
Prepare the raw data for model training by preprocessing with splitting the data into train and test data.
Further on Feature Scaling for both categorical and numerical feautures. 
"""
# Imports
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path
from logger_config import setup_logger

logger = setup_logger(__name__, log_file="logs/preprocessing.log")

# 1: Data Split Function

def split_data(df, target_col, test_size=0.2, random_state=42):
    """
    Splits the dataset into train and test sets.
    Uses stratified split if the target is categorical.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y 
    )
    logger.info(f"Data split completed: Train={X_train.shape}, Test={X_test.shape}")

    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    logger.info("Train and test splits saved successfully in data/processed/")
    return X_train, X_test, y_train, y_test


# 2: Preprocessing Pipeline

def create_preprocessor(X):
    """
    Creates a ColumnTransformer that scales numerical features and encodes categorical features.
    """
    numeric_features = [
    'LIMIT_BAL', 'AGE',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]

    categorical_features = [
    'SEX', 'EDUCATION', 'MARRIAGE',
     'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'
    ]

    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].astype('category')
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = 'passthrough'

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    logger.info("Preprocessor created successfully.")
    return preprocessor, numeric_features, categorical_features

# 3: Integration 

if __name__ == "__main__":
    try:
        data_path = Path("data/processed/credit_default_cleaned.csv")
        data = pd.read_csv(data_path)
        target_col = 'DEFAULT'
        logger.info(f"Dataset loaded: {data.shape}")

        # Conditional cleanup and renaming 
        if 'ID' in data.columns:
            data = data.drop('ID', axis=1)
            logger.info("'ID' column found and removed.")

        if 'default payment next month' in data.columns:
            data.rename(columns={'default payment next month': 'DEFAULT'}, inplace=True)
            logger.info("Target column renamed: 'default payment next month' → 'DEFAULT'")

        target_col = 'DEFAULT'
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset after renaming.")

        # Split data
        X_train, X_test, y_train, y_test = split_data(data, target_col)
        logger.info(f"Target distribution (train): {y_train.value_counts().to_dict()}")

        # Create and apply preprocessor
        preprocessor, num_cols, cat_cols = create_preprocessor(X_train)
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)

        # Convert to DataFrames
        all_features = np.concatenate([num_cols, cat_cols])
        X_train_preprocessed = pd.DataFrame(X_train_preprocessed, columns=all_features)
        X_test_preprocessed = pd.DataFrame(X_test_preprocessed, columns=all_features)

        # Reset indices
        X_train_preprocessed.reset_index(drop=True, inplace=True)
        X_test_preprocessed.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        # Combine features + target
        train_final = pd.concat([X_train_preprocessed, y_train], axis=1)
        test_final = pd.concat([X_test_preprocessed, y_test], axis=1)

        # Save preprocessed data
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        train_final.to_csv(output_dir / "train_preprocessed.csv", index=False)
        test_final.to_csv(output_dir / "test_preprocessed.csv", index=False)
        logger.info("Preprocessed data saved successfully.")

        # Save fitted preprocessor
        Path("models").mkdir(exist_ok=True)
        joblib.dump(preprocessor, "models/preprocessor.pkl")
        logger.info("Preprocessor saved as models/preprocessor.pkl")
    except Exception as e:
        logger.exception(f"Error during preprocessing: {e}")









