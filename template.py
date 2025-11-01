import os
from pathlib import Path
import logging

# Configure logging format
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# Project name
project_name = "credit_default_explainer"

# List of all required files and directories
list_of_files = [
    # GitHub workflow
    ".github/workflows/.gitkeep",

    # Source code structure
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/data_preprocessing.py",
    f"src/{project_name}/model_training.py",
    f"src/{project_name}/explainability.py",

    # Main execution file
    "main.py",

    # Data directories
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",

    # Model directory
    "models/.gitkeep",

    # Reports and results
    "reports/figures/.gitkeep",
    "reports/findings.md",

    # Documentation
    "docs/data_description.md",

    # Configuration and dependencies
    "requirements.txt",
    ".gitignore",
    "README.md",

    # Notebook directory for exploration
    "notebooks/01_eda.ipynb",
    "notebooks/02_model_training.ipynb",
    "notebooks/03_explainability.ipynb",

    # Streamlit app
    "app/app.py",
]

# Create directories and files
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Created empty file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")
