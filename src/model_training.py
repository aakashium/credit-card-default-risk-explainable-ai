"""
model_training.py
Train and evaluate a classification model using Random Undersampling and XGBoost.
"""
# Imports
import os
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from joblib import dump
from logger_config import setup_logger

# Initialize logger
logger = setup_logger("model_training_logger", "logs/model_training.log")

# Random Underscoring Function

def apply_random_undersampling(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """
    Balances the dataset using RandomUnderSampler from imblearn.
    """
    logger.info("Applying random undersampling...")
    rus = RandomUnderSampler(random_state=random_state)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    logger.info(f"Resampling complete. New class distribution:\n{y_resampled.value_counts().to_dict()}")
    return X_resampled, y_resampled


# Model Training Function

def train_model(
    TRAIN_PATH: str,
    TEST_PATH: str,
    TARGET_COL: str,
    model_save_path: str = "./models/trained_model.joblib"
):
    """
    Loads preprocessed train/test CSVs, applies random undersampling,
    trains a classifier, evaluates it, and saves the model.
    """
    try:
        # 1. Load data
        logger.info("Loading preprocessed data...")
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
        logger.info(f"Loaded {len(train_df)} training samples and {len(test_df)} testing samples.")

        X_train = train_df.drop(columns=[TARGET_COL])
        y_train = train_df[TARGET_COL]
        X_test = test_df.drop(columns=[TARGET_COL])
        y_test = test_df[TARGET_COL]

        logger.info(f"Class distribution before undersampling: {y_train.value_counts().to_dict()}")

        # 2. Apply undersampling
        X_train_bal, y_train_bal = apply_random_undersampling(X_train, y_train)

        # 3. Define model
        logger.info("Initializing XGBoost classifier...")
        model = XGBClassifier(
            learning_rate=0.1,
            max_depth=7,
            subsample=0.8,
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            eval_metric="auc",
            use_label_encoder=False
        )

        # 4. Train model
        logger.info("Training model...")
        model.fit(X_train_bal, y_train_bal)
        logger.info("Model training completed.")

        # 5. Evaluate model
        logger.info("Evaluating model performance...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics = {
        "Accuracy": float(acc),
        "Recall": float(rec),
        "F1 Score": float(f1),
        "ROC-AUC": float(auc)
        }

        logger.info(f"Evaluation Results: {metrics}")

        # 6. Save model
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        dump(model, model_save_path)
        logger.info(f"Model saved successfully at: {model_save_path}")

        return model, metrics

    except Exception as e:
        logger.exception(f"Error during model training: {str(e)}")
        raise


if __name__ == "__main__":
    logger.info("Starting Model Training Pipeline")

    TRAIN_PATH = "./data/processed/train_preprocessed.csv"
    TEST_PATH = "./data/processed/test_preprocessed.csv"
    TARGET_COL = "DEFAULT"

    trained_model = train_model(TRAIN_PATH, TEST_PATH, TARGET_COL)

    logger.info("Model Training Completed Successfully")
