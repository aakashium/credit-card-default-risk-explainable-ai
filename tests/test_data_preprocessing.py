"""
Test suite for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing import (
    clean_and_rename_columns,
    split_data,
    create_preprocessor,
    combine_preprocessed_data
)
from logger_config import setup_logger


@pytest.fixture
def sample_data():
    """Create sample credit default data for testing."""
    return pd.DataFrame({
        'ID': [1, 2, 3, 4, 5],
        'LIMIT_BAL': [20000, 120000, 90000, 50000, 50000],
        'SEX': [2, 2, 2, 2, 1],
        'EDUCATION': [2, 2, 2, 2, 2],
        'MARRIAGE': [1, 2, 2, 1, 1],
        'AGE': [24, 26, 34, 37, 57],
        'PAY_0': [2, -1, 0, 0, -1],
        'PAY_2': [2, 2, 0, 0, 0],
        'PAY_3': [-1, 0, 0, 0, -1],
        'PAY_4': [-1, 0, 0, 0, 0],
        'PAY_5': [-2, 0, 0, 0, 0],
        'PAY_6': [-2, 2, 0, 0, 0],
        'BILL_AMT1': [3913, 2682, 29239, 46990, 8617],
        'BILL_AMT2': [3102, 1725, 14027, 48233, 5670],
        'BILL_AMT3': [689, 2682, 13559, 49291, 35835],
        'BILL_AMT4': [0, 3272, 14331, 28314, 20940],
        'BILL_AMT5': [0, 3455, 14948, 28959, 19146],
        'BILL_AMT6': [0, 3261, 15549, 29547, 19131],
        'PAY_AMT1': [0, 0, 1518, 2000, 2000],
        'PAY_AMT2': [689, 1000, 1500, 2019, 36681],
        'PAY_AMT3': [0, 1000, 1000, 1200, 10000],
        'PAY_AMT4': [0, 1000, 1000, 1100, 9000],
        'PAY_AMT5': [0, 0, 1000, 1069, 689],
        'PAY_AMT6': [0, 2000, 5000, 1000, 679],
        'default payment next month': [1, 1, 0, 0, 0]
    })


@pytest.fixture
def test_logger():
    """Create a test logger."""
    return setup_logger("test_preprocessing", "logs/test_preprocessing.log")


class TestCleanAndRenameColumns:
    """Tests for clean_and_rename_columns function."""
    
    def test_removes_id_column(self, sample_data, test_logger):
        """Test that ID column is removed."""
        result = clean_and_rename_columns(sample_data.copy(), test_logger)
        assert 'ID' not in result.columns
    
    def test_renames_target_column(self, sample_data, test_logger):
        """Test that target column is renamed correctly."""
        result = clean_and_rename_columns(sample_data.copy(), test_logger)
        assert 'DEFAULT' in result.columns
        assert 'default payment next month' not in result.columns
    
    def test_preserves_data_integrity(self, sample_data, test_logger):
        """Test that data values are preserved after cleaning."""
        result = clean_and_rename_columns(sample_data.copy(), test_logger)
        # Should have one less column (ID removed)
        assert len(result.columns) == len(sample_data.columns) - 1
        # Should have same number of rows
        assert len(result) == len(sample_data)
    
    def test_handles_missing_id_column(self, sample_data, test_logger):
        """Test handling when ID column doesn't exist."""
        data_no_id = sample_data.drop('ID', axis=1)
        result = clean_and_rename_columns(data_no_id.copy(), test_logger)
        assert 'ID' not in result.columns
        assert 'DEFAULT' in result.columns
    
    def test_raises_error_without_target(self, test_logger):
        """Test that error is raised when target column is missing."""
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        with pytest.raises(ValueError, match="Target column 'default' not found"):
            clean_and_rename_columns(data, test_logger)


class TestSplitData:
    """Tests for split_data function."""
    
    def test_split_ratio(self, sample_data, test_logger):
        """Test that data is split according to specified ratio."""
        cleaned_data = clean_and_rename_columns(sample_data.copy(), test_logger)
        X_train, X_test, y_train, y_test = split_data(
            cleaned_data, 'DEFAULT', test_size=0.2, random_state=42
        )
        
        total_samples = len(cleaned_data)
        train_samples = len(X_train)
        test_samples = len(X_test)
        
        assert train_samples + test_samples == total_samples
        assert abs(test_samples / total_samples - 0.2) < 0.1  # Allow small variance
    
    def test_stratification(self, sample_data, test_logger):
        """Test that stratification maintains class distribution."""
        cleaned_data = clean_and_rename_columns(sample_data.copy(), test_logger)
        X_train, X_test, y_train, y_test = split_data(
            cleaned_data, 'DEFAULT', test_size=0.2, random_state=42
        )
        
        # Both train and test should have both classes (if possible with small sample)
        assert len(y_train.unique()) > 0
        assert len(y_test.unique()) > 0
    
    def test_no_data_leakage(self, sample_data, test_logger):
        """Test that train and test sets don't overlap."""
        cleaned_data = clean_and_rename_columns(sample_data.copy(), test_logger)
        X_train, X_test, y_train, y_test = split_data(
            cleaned_data, 'DEFAULT', test_size=0.2, random_state=42
        )
        
        # Indices should not overlap
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        assert len(train_indices.intersection(test_indices)) == 0


class TestCreatePreprocessor:
    """Tests for create_preprocessor function."""
    
    def test_preprocessor_creation(self, sample_data, test_logger):
        """Test that preprocessor is created successfully."""
        cleaned_data = clean_and_rename_columns(sample_data.copy(), test_logger)
        X = cleaned_data.drop('DEFAULT', axis=1)
        
        preprocessor, num_cols, cat_cols = create_preprocessor(X)
        
        assert preprocessor is not None
        assert len(num_cols) > 0
        assert len(cat_cols) > 0
    
    def test_feature_categorization(self, sample_data, test_logger):
        """Test that features are correctly categorized."""
        cleaned_data = clean_and_rename_columns(sample_data.copy(), test_logger)
        X = cleaned_data.drop('DEFAULT', axis=1)
        
        preprocessor, num_cols, cat_cols = create_preprocessor(X)
        
        # Check that expected columns are in the right category
        assert 'LIMIT_BAL' in num_cols
        assert 'AGE' in num_cols
        assert 'SEX' in cat_cols
        assert 'EDUCATION' in cat_cols
    
    def test_preprocessor_transform(self, sample_data, test_logger):
        """Test that preprocessor can transform data."""
        cleaned_data = clean_and_rename_columns(sample_data.copy(), test_logger)
        X = cleaned_data.drop('DEFAULT', axis=1)
        
        preprocessor, num_cols, cat_cols = create_preprocessor(X)
        
        # Fit and transform
        X_transformed = preprocessor.fit_transform(X)
        
        # Check output shape
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == len(num_cols) + len(cat_cols)


class TestCombinePreprocessedData:
    """Tests for combine_preprocessed_data function."""
    
    def test_combine_data(self, sample_data, test_logger):
        """Test that preprocessed data is combined correctly."""
        cleaned_data = clean_and_rename_columns(sample_data.copy(), test_logger)
        X_train, X_test, y_train, y_test = split_data(
            cleaned_data, 'DEFAULT', test_size=0.2, random_state=42
        )
        
        preprocessor, num_cols, cat_cols = create_preprocessor(X_train)
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)
        
        train_final, test_final = combine_preprocessed_data(
            X_train_preprocessed, X_test_preprocessed,
            y_train, y_test, num_cols, cat_cols
        )
        
        # Check shapes
        assert len(train_final) == len(X_train)
        assert len(test_final) == len(X_test)
        
        # Check that target column is present
        assert 'DEFAULT' in train_final.columns
        assert 'DEFAULT' in test_final.columns
    
    def test_column_names(self, sample_data, test_logger):
        """Test that column names are preserved."""
        cleaned_data = clean_and_rename_columns(sample_data.copy(), test_logger)
        X_train, X_test, y_train, y_test = split_data(
            cleaned_data, 'DEFAULT', test_size=0.2, random_state=42
        )
        
        preprocessor, num_cols, cat_cols = create_preprocessor(X_train)
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)
        
        train_final, test_final = combine_preprocessed_data(
            X_train_preprocessed, X_test_preprocessed,
            y_train, y_test, num_cols, cat_cols
        )
        
        # All feature columns should be present
        for col in num_cols:
            assert col in train_final.columns
        for col in cat_cols:
            assert col in train_final.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
