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
from src.utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__, log_file="logs/preprocessing.log")


# 1 Clean and rename columns
def clean_and_rename_columns(data, logger):
    """
    Cleans up and renames specific columns in the dataset.
    
    This function performs the following operations:
    1. Removes the 'ID' column if present (not needed for modeling)
    2. Renames 'default payment next month' to 'DEFAULT' for consistency
    3. Validates that the target column exists after renaming
    
    Args:
        data (pd.DataFrame): Input DataFrame to clean and rename.
        logger (logging.Logger): Logger instance for tracking operations.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with renamed columns.
    
    Raises:
        TypeError: If data is not a pandas DataFrame.
        ValueError: If target column 'DEFAULT' is not found after renaming.
        ValueError: If DataFrame is empty.
    
    Example:
        >>> df = pd.DataFrame({
        ...     'ID': [1, 2, 3],
        ...     'default payment next month': [0, 1, 0],
        ...     'LIMIT_BAL': [20000, 50000, 30000]
        ... })
        >>> cleaned_df = clean_and_rename_columns(df, logger)
        >>> 'ID' in cleaned_df.columns
        False
        >>> 'DEFAULT' in cleaned_df.columns
        True
    """
    # Input validation
    if not isinstance(data, pd.DataFrame):
        error_msg = f"Expected pandas DataFrame, got {type(data).__name__}"
        logger.error(error_msg)
        raise TypeError(error_msg)
    
    if data.empty:
        error_msg = "Cannot process empty DataFrame"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Starting column cleanup. Input shape: {data.shape}")
    
    # Remove ID column if present
    if 'ID' in data.columns: 
        data = data.drop('ID', axis=1) 
        logger.info("'ID' column found and removed.") 
    else: 
        logger.info("'ID' column not found — skipping removal.") 
    
    # Rename target column
    if 'default payment next month' in data.columns: 
        data.rename(columns={'default payment next month': 'DEFAULT'}, inplace=True) 
        logger.info("Target column renamed: 'default payment next month' → 'DEFAULT'") 
    else: 
        logger.info("'default payment next month' column not found — skipping renaming.") 
    
    # Validate target column exists
    if 'DEFAULT' not in data.columns:
        error_msg = (
            "Target column 'DEFAULT' not found after cleanup. "
            f"Available columns: {list(data.columns)}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Column cleanup completed successfully. Output shape: {data.shape}")
    return data


# 2 Split data into train and test
def split_data(df, target_col, test_size=0.2, random_state=42):
    """
    Split dataset into training and testing sets with stratification.
    
    Performs stratified train-test split to maintain class distribution
    in both sets, which is important for imbalanced datasets.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing features and target.
        target_col (str): Name of the target column.
        test_size (float, optional): Proportion of dataset for testing.
            Must be between 0 and 1. Defaults to 0.2 (20%).
        random_state (int, optional): Random seed for reproducibility.
            Defaults to 42.
    
    Returns:
        tuple: A tuple containing:
            - X_train (pd.DataFrame): Training features
            - X_test (pd.DataFrame): Testing features
            - y_train (pd.Series): Training labels
            - y_test (pd.Series): Testing labels
    
    Raises:
        ValueError: If target_col not in DataFrame columns.
        ValueError: If test_size is not between 0 and 1.
        ValueError: If DataFrame has insufficient samples for splitting.
    
    Example:
        >>> df = pd.DataFrame({
        ...     'feature1': [1, 2, 3, 4, 5],
        ...     'feature2': [5, 4, 3, 2, 1],
        ...     'target': [0, 1, 0, 1, 0]
        ... })
        >>> X_train, X_test, y_train, y_test = split_data(df, 'target')
        >>> len(X_train) + len(X_test) == len(df)
        True
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        error_msg = f"Expected pandas DataFrame, got {type(df).__name__}"
        logger.error(error_msg)
        raise TypeError(error_msg)
    
    if target_col not in df.columns:
        error_msg = (
            f"Target column '{target_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if not 0 < test_size < 1:
        error_msg = f"test_size must be between 0 and 1, got {test_size}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    min_samples = 10  # Minimum samples needed for meaningful split
    if len(df) < min_samples:
        error_msg = (
            f"DataFrame has only {len(df)} samples. "
            f"Need at least {min_samples} for train-test split."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    except ValueError as e:
        # Stratification might fail if class has too few samples
        logger.warning(f"Stratified split failed: {e}. Attempting non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=None
        )

    logger.info(f"Data split completed: Train={X_train.shape}, Test={X_test.shape}")
    logger.info(f"Train class distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"Test class distribution: {y_test.value_counts().to_dict()}")
    
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
    """
    Combine preprocessed features with target labels into final DataFrames.
    
    This function:
    1. Converts numpy arrays back to DataFrames with proper column names
    2. Resets indices to ensure proper alignment
    3. Concatenates features and labels
    
    Args:
        X_train_preprocessed (np.ndarray): Preprocessed training features.
        X_test_preprocessed (np.ndarray): Preprocessed testing features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
        num_cols (list): List of numerical column names.
        cat_cols (list): List of categorical column names.
    
    Returns:
        tuple: A tuple containing:
            - train_final (pd.DataFrame): Training data with features and target
            - test_final (pd.DataFrame): Testing data with features and target
    
    Raises:
        ValueError: If shapes don't match between features and labels.
        ValueError: If column names don't match array dimensions.
    
    Example:
        >>> X_train = np.array([[1, 2, 3], [4, 5, 6]])
        >>> X_test = np.array([[7, 8, 9]])
        >>> y_train = pd.Series([0, 1])
        >>> y_test = pd.Series([0])
        >>> num_cols = ['A', 'B']
        >>> cat_cols = ['C']
        >>> train_df, test_df = combine_preprocessed_data(
        ...     X_train, X_test, y_train, y_test, num_cols, cat_cols
        ... )
        >>> len(train_df) == len(y_train)
        True
    """
    # Input validation
    if len(X_train_preprocessed) != len(y_train):
        error_msg = (
            f"Training features ({len(X_train_preprocessed)}) and labels ({len(y_train)}) "
            "have different lengths"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if len(X_test_preprocessed) != len(y_test):
        error_msg = (
            f"Testing features ({len(X_test_preprocessed)}) and labels ({len(y_test)}) "
            "have different lengths"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Combine column names
    all_features = np.concatenate([num_cols, cat_cols])
    
    # Validate column count matches
    if X_train_preprocessed.shape[1] != len(all_features):
        error_msg = (
            f"Number of columns in preprocessed data ({X_train_preprocessed.shape[1]}) "
            f"doesn't match number of feature names ({len(all_features)})"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("Combining preprocessed features with labels")
    
    # Convert to DataFrames
    X_train_preprocessed = pd.DataFrame(X_train_preprocessed, columns=all_features)
    X_test_preprocessed = pd.DataFrame(X_test_preprocessed, columns=all_features)

    # Reset indices for proper concatenation
    X_train_preprocessed.reset_index(drop=True, inplace=True)
    X_test_preprocessed.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # Concatenate features and labels
    train_final = pd.concat([X_train_preprocessed, y_train], axis=1)
    test_final = pd.concat([X_test_preprocessed, y_test], axis=1)

    logger.info(f"Combined data shapes - Train: {train_final.shape}, Test: {test_final.shape}")
    
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
