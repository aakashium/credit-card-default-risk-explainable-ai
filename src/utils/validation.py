"""
validation.py
-------------
Data validation utilities for ensuring data quality and detecting issues.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging


@dataclass
class ValidationReport:
    """Container for validation results."""
    
    passed: bool
    errors: List[str]
    warnings: List[str]
    info: Dict[str, Any]
    
    def __str__(self) -> str:
        """String representation of validation report."""
        lines = [
            "=" * 60,
            "VALIDATION REPORT",
            "=" * 60,
            f"Status: {'✓ PASSED' if self.passed else '✗ FAILED'}",
            ""
        ]
        
        if self.errors:
            lines.append("ERRORS:")
            for error in self.errors:
                lines.append(f"  ✗ {error}")
            lines.append("")
        
        if self.warnings:
            lines.append("WARNINGS:")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")
            lines.append("")
        
        if self.info:
            lines.append("INFO:")
            for key, value in self.info.items():
                lines.append(f"  • {key}: {value}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


def validate_dataframe_schema(
    df: pd.DataFrame,
    required_columns: List[str],
    optional_columns: Optional[List[str]] = None,
    column_types: Optional[Dict[str, type]] = None
) -> ValidationReport:
    """
    Validate DataFrame schema against expected structure.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        optional_columns: List of optional column names
        column_types: Expected data types for columns
        
    Returns:
        ValidationReport with results
    """
    errors = []
    warnings = []
    info = {}
    
    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    # Check for unexpected columns
    expected_cols = set(required_columns)
    if optional_columns:
        expected_cols.update(optional_columns)
    
    unexpected_cols = set(df.columns) - expected_cols
    if unexpected_cols:
        warnings.append(f"Unexpected columns found: {unexpected_cols}")
    
    # Check column types
    if column_types:
        for col, expected_type in column_types.items():
            if col in df.columns:
                actual_type = df[col].dtype
                if not pd.api.types.is_dtype_equal(actual_type, expected_type):
                    warnings.append(
                        f"Column '{col}' has type {actual_type}, expected {expected_type}"
                    )
    
    info['total_columns'] = len(df.columns)
    info['total_rows'] = len(df)
    
    return ValidationReport(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        info=info
    )


def check_missing_values(
    df: pd.DataFrame,
    threshold: float = 0.5,
    critical_columns: Optional[List[str]] = None
) -> ValidationReport:
    """
    Check for missing values in DataFrame.
    
    Args:
        df: DataFrame to check
        threshold: Maximum allowed proportion of missing values (0-1)
        critical_columns: Columns that cannot have any missing values
        
    Returns:
        ValidationReport with results
    """
    errors = []
    warnings = []
    info = {}
    
    # Calculate missing value statistics
    missing_counts = df.isnull().sum()
    missing_pct = missing_counts / len(df)
    
    # Check critical columns
    if critical_columns:
        for col in critical_columns:
            if col in df.columns and missing_counts[col] > 0:
                errors.append(
                    f"Critical column '{col}' has {missing_counts[col]} missing values"
                )
    
    # Check threshold
    high_missing = missing_pct[missing_pct > threshold]
    if not high_missing.empty:
        for col, pct in high_missing.items():
            warnings.append(
                f"Column '{col}' has {pct*100:.1f}% missing values (threshold: {threshold*100:.1f}%)"
            )
    
    # Info
    total_missing = missing_counts.sum()
    info['total_missing_values'] = int(total_missing)
    info['missing_percentage'] = f"{(total_missing / (len(df) * len(df.columns)) * 100):.2f}%"
    info['columns_with_missing'] = list(missing_counts[missing_counts > 0].index)
    
    return ValidationReport(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        info=info
    )


def check_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first'
) -> ValidationReport:
    """
    Check for duplicate rows in DataFrame.
    
    Args:
        df: DataFrame to check
        subset: Columns to consider for identifying duplicates
        keep: Which duplicates to mark ('first', 'last', False)
        
    Returns:
        ValidationReport with results
    """
    errors = []
    warnings = []
    info = {}
    
    # Find duplicates
    duplicates = df.duplicated(subset=subset, keep=keep)
    num_duplicates = duplicates.sum()
    
    if num_duplicates > 0:
        warnings.append(f"Found {num_duplicates} duplicate rows")
        
        # Get duplicate indices
        duplicate_indices = df[duplicates].index.tolist()
        info['duplicate_indices'] = duplicate_indices[:10]  # First 10
        if len(duplicate_indices) > 10:
            info['note'] = f"Showing first 10 of {len(duplicate_indices)} duplicates"
    
    info['total_duplicates'] = int(num_duplicates)
    info['duplicate_percentage'] = f"{(num_duplicates / len(df) * 100):.2f}%"
    
    return ValidationReport(
        passed=True,  # Duplicates are warnings, not errors
        errors=errors,
        warnings=warnings,
        info=info
    )


def check_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'iqr',
    threshold: float = 3.0
) -> ValidationReport:
    """
    Check for outliers in numerical columns.
    
    Args:
        df: DataFrame to check
        columns: Columns to check (None = all numeric columns)
        method: Outlier detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        ValidationReport with results
    """
    errors = []
    warnings = []
    info = {}
    
    # Select numeric columns
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_summary = {}
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = z_scores > threshold
        else:
            errors.append(f"Unknown outlier detection method: {method}")
            continue
        
        num_outliers = outliers.sum()
        if num_outliers > 0:
            outlier_pct = (num_outliers / len(df)) * 100
            outlier_summary[col] = {
                'count': int(num_outliers),
                'percentage': f"{outlier_pct:.2f}%"
            }
            
            if outlier_pct > 10:  # More than 10% outliers
                warnings.append(
                    f"Column '{col}' has {num_outliers} outliers ({outlier_pct:.1f}%)"
                )
    
    info['outlier_summary'] = outlier_summary
    info['detection_method'] = method
    info['threshold'] = threshold
    
    return ValidationReport(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        info=info
    )


def check_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    threshold: float = 0.05
) -> ValidationReport:
    """
    Detect data drift between reference and current datasets.
    
    Uses Kolmogorov-Smirnov test for numerical columns.
    
    Args:
        reference_df: Reference (training) dataset
        current_df: Current (production) dataset
        columns: Columns to check (None = all common numeric columns)
        threshold: P-value threshold for drift detection
        
    Returns:
        ValidationReport with results
    """
    from scipy.stats import ks_2samp
    
    errors = []
    warnings = []
    info = {}
    
    # Find common numeric columns
    if columns is None:
        ref_numeric = set(reference_df.select_dtypes(include=[np.number]).columns)
        cur_numeric = set(current_df.select_dtypes(include=[np.number]).columns)
        columns = list(ref_numeric.intersection(cur_numeric))
    
    drift_detected = {}
    
    for col in columns:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        
        # Perform KS test
        statistic, p_value = ks_2samp(
            reference_df[col].dropna(),
            current_df[col].dropna()
        )
        
        drift_detected[col] = {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'drift': p_value < threshold
        }
        
        if p_value < threshold:
            warnings.append(
                f"Data drift detected in column '{col}' (p-value: {p_value:.4f})"
            )
    
    info['drift_summary'] = drift_detected
    info['threshold'] = threshold
    info['columns_with_drift'] = [
        col for col, stats in drift_detected.items() if stats['drift']
    ]
    
    return ValidationReport(
        passed=True,  # Drift is a warning, not an error
        errors=errors,
        warnings=warnings,
        info=info
    )


def comprehensive_validation(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    critical_columns: Optional[List[str]] = None,
    missing_threshold: float = 0.5,
    check_for_duplicates: bool = True,
    check_for_outliers: bool = True
) -> Dict[str, ValidationReport]:
    """
    Run comprehensive validation checks on a DataFrame.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required columns
        critical_columns: Columns that cannot have missing values
        missing_threshold: Maximum allowed proportion of missing values
        check_for_duplicates: Whether to check for duplicates
        check_for_outliers: Whether to check for outliers
        
    Returns:
        Dictionary of validation reports
    """
    reports = {}
    
    # Schema validation
    if required_columns:
        reports['schema'] = validate_dataframe_schema(df, required_columns)
    
    # Missing values
    reports['missing_values'] = check_missing_values(
        df,
        threshold=missing_threshold,
        critical_columns=critical_columns
    )
    
    # Duplicates
    if check_for_duplicates:
        reports['duplicates'] = check_duplicates(df)
    
    # Outliers
    if check_for_outliers:
        reports['outliers'] = check_outliers(df)
    
    return reports


if __name__ == "__main__":
    # Example usage
    # Create sample data with issues
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10],  # Has outlier
        'B': [1, 2, None, 4, 5, 6, 7, 8, 9, 10],  # Has missing
        'C': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
        'D': [1, 2, 3, 4, 5, 6, 7, 8, 9, 1]  # Has duplicate
    })
    
    # Run comprehensive validation
    reports = comprehensive_validation(
        df,
        required_columns=['A', 'B', 'C'],
        critical_columns=['A']
    )
    
    # Print reports
    for check_name, report in reports.items():
        print(f"\n{check_name.upper()}:")
        print(report)
