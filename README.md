# Explainable Credit Risk Prediction

A modular Machine Learning pipeline for predicting credit default risk with a focus on **Explainable AI (XAI)** using SHAP. This project includes data preprocessing, model training with **XGBoost**, and an interactive **Streamlit** dashboard.

## 🚀 Features

*   **Modular Pipeline**: Clean separation of Data Preprocessing, Model Training, and Explainability.
*   **Best Practices**: Structured logging, configuration management, and type hinting.
*   **Explainable AI**: Integrated SHAP support for global (feature importance) and local (individual prediction) explanations.
*   **Interactive Dashboard**: Streamlit app to control the pipeline and visualize results.
*   **Data Validation**: Robust checks for data integrity, missing values, and drift.
*   **Containerized**: Docker support for easy deployment.

## 📁 Project Structure

```
Explainable-credit-risk-prediction/
├── data/                   # Data directory
│   ├── raw/                # Original data
│   └── processed/          # Cleaned and splitted data
├── notebooks/              # Jupyter Notebooks for experiments
│   ├── 01_eda.ipynb        # Exploratory Data Analysis
│   └── 04_feature_engineering_experiment.ipynb # Feature Engineering & RFE
├── src/                    # Source code
│   ├── utils/              # Utilities (Logger, Config, Validation)
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── explainability.py
├── tests/                  # Unit tests
├── logs/                   # Application logs
├── app.py                  # Streamlit Dashboard Entrypoint
├── requirements.txt        # Production dependencies
├── Dockerfile              # Docker configuration
└── README.md               # Project documentation
```

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/aakashium/credit-card-default-risk-explainable-ai.git
    cd credit-card-default-risk-explainable-ai
    ```

2.  **Set up the environment (using uv is recommended):**
    ```bash
    # Create virtual environment
    uv venv
    
    # Activate validation
    # Windows:
    .venv\Scripts\activate
    # Linux/Mac:
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```

## 🏃‍♂️ How to Run

### 1. Interactive Dashboard (Streamlit)
The easiest way to run the full pipeline is via the dashboard.

```bash
streamlit run app.py
```
This will open a browser window where you can:
*   Run Data Preprocessing
*   Train the Model
*   Generate SHAP Explanations

### 2. Feature Engineering Experiments
To run the advanced feature engineering and RFE experiments:

```bash
jupyter notebook notebooks/04_feature_engineering_experiment.ipynb
```
Or execute it directly via command line:
```bash
jupyter nbconvert --to notebook --execute notebooks/04_feature_engineering_experiment.ipynb
```

### 3. Docker
To run using Docker:

```bash
# Build image
docker build -t credit-risk-app .

# Run container
docker run -p 8501:8501 credit-risk-app
```

## 🧪 Development

*   **Run Implemented Tests:**
    ```bash
    pytest tests/
    ```

*   **Code Quality Checks:**
    ```bash
    pre-commit run --all-files
    ```

## 📝 Configuration
Project settings (paths, hyperparameters) are managed in `src/utils/config.py`.

## 🤝 Contributing
1.  Fork the repository
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
