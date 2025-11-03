"""
data_preprocessing.py
---------------------
Prepare the raw data for model training by preprocessing, splitting into train and test sets,
and performing feature scaling for both categorical and numerical features.
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

# Initialize logger
logger = setup_logger(__name__, log_file="logs/preprocessing.log")


# 1 Clean and rename columns
def clean_and_rename_columns(data, logger): 
    """Cleans up and renames specific columns in the dataset.""" 
    if 'ID' in data.columns: 
        data = data.drop('ID', axis=1) 
        logger.info("'ID' column found and removed.") 
    else: 
        logger.info("'ID' column not found — skipping removal.") 
    
    if 'default payment next month' in data.columns: 
        data.rename(columns={'default payment next month': 'DEFAULT'}, inplace=True) 
        logger.info("Target column renamed: 'default payment next month' → 'DEFAULT'") 
    else: 
        logger.info("'default payment next month' column not found — skipping renaming.") 
        
    if 'DEFAULT' not in data.columns:
        logger.error("Target column 'default' not found after cleanup.")
        raise ValueError("Target column 'default' not found after renaming.")

    logger.info("Column cleanup and renaming completed successfully.")
    return data


# 2 Split data into train and test
def split_data(df, target_col, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info(f"Data split completed: Train={X_train.shape}, Test={X_test.shape}")
    return X_train, X_test, y_train, y_test


# 3 Create preprocessing pipeline
def create_preprocessor(X):
    numeric_features = [
        'LIMIT_BAL', 'AGE',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
        'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
        'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]

    categorical_features = [
        'SEX', 'EDUCATION', 'MARRIAGE',
        'PAY_0', 'PAY_2', 'PAY_3',
        'PAY_4', 'PAY_5', 'PAY_6'
    ]

    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].astype('category')

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', 'passthrough', categorical_features)
        ]
    )

    logger.info("Preprocessor created successfully.")
    return preprocessor, numeric_features, categorical_features


# 4 Combine preprocessed data
def combine_preprocessed_data(X_train_preprocessed, X_test_preprocessed, y_train, y_test, num_cols, cat_cols):
    all_features = np.concatenate([num_cols, cat_cols])
    X_train_preprocessed = pd.DataFrame(X_train_preprocessed, columns=all_features)
    X_test_preprocessed = pd.DataFrame(X_test_preprocessed, columns=all_features)

    X_train_preprocessed.reset_index(drop=True, inplace=True)
    X_test_preprocessed.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    train_final = pd.concat([X_train_preprocessed, y_train], axis=1)
    test_final = pd.concat([X_test_preprocessed, y_test], axis=1)

    return train_final, test_final


# 5 Save outputs
def save_preprocessed_outputs(train_final, test_final, preprocessor, logger,
                              processed_dir="data/processed", model_dir="models"):
    output_dir = Path(processed_dir)
    model_path = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_preprocessed.csv"
    test_path = output_dir / "test_preprocessed.csv"
    train_final.to_csv(train_path, index=False)
    test_final.to_csv(test_path, index=False)
    logger.info(f"Preprocessed data saved: {train_path}, {test_path}")

    preprocessor_path = model_path / "preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Preprocessor saved successfully at {preprocessor_path}")

    return train_path, test_path, preprocessor_path


# 6 Final Function
def run_data_preprocessing(
    raw_data_path="data/processed/credit_default_cleaned.csv",
    processed_dir="data/processed",
    model_dir="models"
):
    """
    End-to-end preprocessing function.
    Cleans, splits, scales, and saves the processed datasets.
    Returns file paths for train/test/preprocessor.
    """

    try:
        data_path = Path(raw_data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        data = pd.read_csv(data_path)
        logger.info(f"Dataset loaded successfully with shape: {data.shape}")

        # Step 1: Clean and rename
        data = clean_and_rename_columns(data, logger)
        target_col = 'DEFAULT'

        # Step 2: Split
        X_train, X_test, y_train, y_test = split_data(data, target_col)
        logger.info(f"Target distribution (train): {y_train.value_counts().to_dict()}")

        # Step 3: Preprocessor
        preprocessor, num_cols, cat_cols = create_preprocessor(X_train)
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)

        # Step 4: Combine
        train_final, test_final = combine_preprocessed_data(
            X_train_preprocessed, X_test_preprocessed,
            y_train, y_test, num_cols, cat_cols
        )

        # Step 5: Save
        train_path, test_path, preprocessor_path = save_preprocessed_outputs(
            train_final, test_final, preprocessor, logger,
            processed_dir, model_dir
        )

        logger.info("Data preprocessing pipeline executed successfully.")
        return train_path, test_path, preprocessor_path

    except Exception as e:
        logger.exception(f"Error during preprocessing: {e}")
        raise


# Main()
if __name__ == "__main__":
    run_data_preprocessing()
