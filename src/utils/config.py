"""
config.py
---------
Centralized configuration for the credit risk prediction project.
Contains all paths, hyperparameters, and settings used across modules.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class PathConfig:
    """Configuration for file paths and directories."""
    
    # Base directories
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    MODEL_DIR: Path = BASE_DIR / "models"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    FIGURES_DIR: Path = REPORTS_DIR / "figures"
    LOGS_DIR: Path = BASE_DIR / "logs"
    DOCS_DIR: Path = BASE_DIR / "docs"
    
    # Data files
    RAW_DATA_FILE: str = "credit_default_cleaned.csv"
    TRAIN_FILE: str = "train_preprocessed.csv"
    TEST_FILE: str = "test_preprocessed.csv"
    
    # Model files
    TRAINED_MODEL_FILE: str = "trained_model.joblib"
    PREPROCESSOR_FILE: str = "preprocessor.pkl"
    
    # Log files
    APP_LOG: str = "app.log"
    PREPROCESSING_LOG: str = "preprocessing.log"
    MODEL_TRAINING_LOG: str = "model_training.log"
    EXPLAINABILITY_LOG: str = "explainability.log"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for dir_path in [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.MODEL_DIR,
            self.REPORTS_DIR,
            self.FIGURES_DIR,
            self.LOGS_DIR,
            self.DOCS_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def raw_data_path(self) -> Path:
        """Full path to raw data file."""
        return self.PROCESSED_DATA_DIR / self.RAW_DATA_FILE
    
    @property
    def train_path(self) -> Path:
        """Full path to training data file."""
        return self.PROCESSED_DATA_DIR / self.TRAIN_FILE
    
    @property
    def test_path(self) -> Path:
        """Full path to test data file."""
        return self.PROCESSED_DATA_DIR / self.TEST_FILE
    
    @property
    def model_path(self) -> Path:
        """Full path to trained model file."""
        return self.MODEL_DIR / self.TRAINED_MODEL_FILE
    
    @property
    def preprocessor_path(self) -> Path:
        """Full path to preprocessor file."""
        return self.MODEL_DIR / self.PREPROCESSOR_FILE
    
    @property
    def app_log_path(self) -> Path:
        """Full path to app log file."""
        return self.LOGS_DIR / self.APP_LOG
    
    @property
    def preprocessing_log_path(self) -> Path:
        """Full path to preprocessing log file."""
        return self.LOGS_DIR / self.PREPROCESSING_LOG
    
    @property
    def model_training_log_path(self) -> Path:
        """Full path to model training log file."""
        return self.LOGS_DIR / self.MODEL_TRAINING_LOG
    
    @property
    def explainability_log_path(self) -> Path:
        """Full path to explainability log file."""
        return self.LOGS_DIR / self.EXPLAINABILITY_LOG


@dataclass
class DataConfig:
    """Configuration for data preprocessing."""
    
    # Target column
    TARGET_COL: str = "DEFAULT"
    
    # Train-test split
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    
    # Feature columns
    NUMERIC_FEATURES: List[str] = field(default_factory=lambda: [
        'LIMIT_BAL', 'AGE',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
        'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
        'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ])
    
    CATEGORICAL_FEATURES: List[str] = field(default_factory=lambda: [
        'SEX', 'EDUCATION', 'MARRIAGE',
        'PAY_0', 'PAY_2', 'PAY_3',
        'PAY_4', 'PAY_5', 'PAY_6'
    ])
    
    # Column renaming
    COLUMN_RENAME_MAP: dict = field(default_factory=lambda: {
        'default payment next month': 'DEFAULT'
    })
    
    # Columns to drop
    COLUMNS_TO_DROP: List[str] = field(default_factory=lambda: ['ID'])


@dataclass
class ModelConfig:
    """Configuration for model training."""
    
    # XGBoost hyperparameters
    LEARNING_RATE: float = 0.1
    MAX_DEPTH: int = 7
    N_ESTIMATORS: int = 300
    SUBSAMPLE: float = 0.8
    RANDOM_STATE: int = 42
    N_JOBS: int = -1
    EVAL_METRIC: str = "auc"
    USE_LABEL_ENCODER: bool = False
    
    # Sampling strategy
    APPLY_UNDERSAMPLING: bool = True
    UNDERSAMPLING_STRATEGY: str = "auto"  # 'auto' or float for custom ratio


@dataclass
class ExplainabilityConfig:
    """Configuration for SHAP explainability."""
    
    # SHAP settings
    USE_TREE_EXPLAINER: bool = True  # Faster for tree-based models
    SAMPLE_SIZE: int = 1000  # Number of samples for SHAP (None = all)
    
    # Plot settings
    TOP_N_FEATURES: int = 5  # Number of features for dependence plots
    SAMPLE_IDX_FOR_LOCAL: int = 10  # Sample index for local explanations
    
    # Figure settings
    FIGURE_DPI: int = 300
    FIGURE_FORMAT: str = "png"


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    
    # Log levels
    LOG_LEVEL: str = "INFO"
    
    # Log format
    LOG_FORMAT: str = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
    
    # Log rotation
    MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
    BACKUP_COUNT: int = 5
    
    # Structured logging
    USE_JSON_LOGGING: bool = False


@dataclass
class Config:
    """Main configuration class combining all sub-configurations."""
    
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Environment
    ENV: str = field(default_factory=lambda: os.getenv("ENV", "development"))
    DEBUG: bool = field(default_factory=lambda: os.getenv("DEBUG", "False").lower() == "true")


# Global configuration instance
config = Config()


# Convenience functions
def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def get_path_config() -> PathConfig:
    """Get path configuration."""
    return config.paths


def get_data_config() -> DataConfig:
    """Get data configuration."""
    return config.data


def get_model_config() -> ModelConfig:
    """Get model configuration."""
    return config.model


def get_explainability_config() -> ExplainabilityConfig:
    """Get explainability configuration."""
    return config.explainability


def get_logging_config() -> LoggingConfig:
    """Get logging configuration."""
    return config.logging


if __name__ == "__main__":
    # Test configuration
    print("=== Path Configuration ===")
    print(f"Base Directory: {config.paths.BASE_DIR}")
    print(f"Raw Data Path: {config.paths.raw_data_path}")
    print(f"Model Path: {config.paths.model_path}")
    
    print("\n=== Data Configuration ===")
    print(f"Target Column: {config.data.TARGET_COL}")
    print(f"Test Size: {config.data.TEST_SIZE}")
    print(f"Numeric Features: {len(config.data.NUMERIC_FEATURES)}")
    
    print("\n=== Model Configuration ===")
    print(f"Learning Rate: {config.model.LEARNING_RATE}")
    print(f"Max Depth: {config.model.MAX_DEPTH}")
    print(f"N Estimators: {config.model.N_ESTIMATORS}")
    
    print("\n=== Explainability Configuration ===")
    print(f"Use Tree Explainer: {config.explainability.USE_TREE_EXPLAINER}")
    print(f"Sample Size: {config.explainability.SAMPLE_SIZE}")
    
    print("\n=== Logging Configuration ===")
    print(f"Log Level: {config.logging.LOG_LEVEL}")
    print(f"Max Bytes: {config.logging.MAX_BYTES}")
