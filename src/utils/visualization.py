"""
visualization.py
----------------
Utility functions for creating model performance visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "Blues"
) -> plt.Figure:
    """
    Plot confusion matrix for binary classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)
        cmap: Colormap for the heatmap
        
    Returns:
        Matplotlib figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=cmap,
        square=True,
        cbar_kws={'label': 'Count'},
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xticklabels(['No Default', 'Default'])
    ax.set_yticklabels(['No Default', 'Default'])
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for positive class
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(
        fpr, tpr,
        color='darkorange',
        lw=2,
        label=f'ROC curve (AUC = {roc_auc:.3f})'
    )
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot Precision-Recall curve for binary classification.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for positive class
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(
        recall, precision,
        color='darkorange',
        lw=2,
        label=f'PR curve (AP = {avg_precision:.3f})'
    )
    ax.axhline(y=y_true.mean(), color='navy', lw=2, linestyle='--', label='Baseline')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance(
    feature_names: list,
    importances: np.ndarray,
    top_n: int = 20,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot feature importance from tree-based models.
    
    Args:
        feature_names: List of feature names
        importances: Feature importance values
        top_n: Number of top features to display
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object
    """
    # Sort features by importance
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    bars = ax.barh(range(len(top_features)), top_importances, color=colors)
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_importances)):
        ax.text(
            value, i,
            f' {value:.4f}',
            va='center',
            fontsize=9
        )
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_class_distribution(
    y: np.ndarray,
    class_names: Optional[list] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot class distribution for classification tasks.
    
    Args:
        y: Target labels
        class_names: Names of classes (optional)
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object
    """
    unique, counts = np.unique(y, return_counts=True)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in unique]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    colors = ['#3498db', '#e74c3c']
    bars = ax1.bar(class_names, counts, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Class Distribution (Count)', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2., height,
            f'{int(count)}\n({count/len(y)*100:.1f}%)',
            ha='center', va='bottom', fontsize=10
        )
    
    # Pie chart
    ax2.pie(
        counts,
        labels=class_names,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        explode=(0.05, 0.05)
    )
    ax2.set_title('Class Distribution (Proportion)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Create visualizations
    plot_confusion_matrix(y_test, y_pred, save_path="reports/figures/test_confusion_matrix.png")
    plot_roc_curve(y_test, y_pred_proba, save_path="reports/figures/test_roc_curve.png")
    plot_precision_recall_curve(y_test, y_pred_proba, save_path="reports/figures/test_pr_curve.png")
    
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    plot_feature_importance(
        feature_names,
        model.feature_importances_,
        save_path="reports/figures/test_feature_importance.png"
    )
    
    plot_class_distribution(
        y_train,
        class_names=['No Default', 'Default'],
        save_path="reports/figures/test_class_distribution.png"
    )
    
    print("Visualization examples created successfully!")
