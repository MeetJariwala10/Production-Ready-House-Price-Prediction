# 🏠 House Price Prediction Project

A comprehensive Machine Learning project that predicts house prices using the Ames Housing dataset. This project demonstrates a complete ML pipeline from data ingestion to model deployment using modern MLOps practices with ZenML and MLflow.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Components](#project-components)
- [Data Pipeline](#data-pipeline)
- [Model Information](#model-information)
- [API Usage](#api-usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

This project implements an end-to-end machine learning pipeline for house price prediction using the Ames Housing dataset. It showcases modern MLOps practices including:

- **Automated Data Processing**: Handles missing values, feature engineering, and outlier detection
- **Model Training**: Linear Regression model with comprehensive preprocessing
- **Experiment Tracking**: MLflow integration for tracking experiments and model versions
- **Model Deployment**: REST API service for real-time predictions
- **Pipeline Orchestration**: ZenML for managing the entire ML workflow

## ✨ Features

- 🔄 **Automated Pipeline**: Complete ML pipeline from data ingestion to deployment
- 📊 **Data Analysis**: Comprehensive EDA with Jupyter notebooks
- 🧹 **Data Preprocessing**: Missing value handling, feature engineering, outlier detection
- 🤖 **Model Training**: Linear Regression with scikit-learn pipeline
- 📈 **Experiment Tracking**: MLflow integration for experiment management
- 🚀 **Model Deployment**: REST API service for predictions
- 📝 **Documentation**: Detailed analysis and code documentation

## 📁 Project Structure

```
House_Price_Prediction/
├── 📊 analysis/                    # Data analysis and EDA
│   ├── EDA.ipynb                  # Jupyter notebook for exploratory analysis
│   └── analysis_src/              # Analysis source code
│       ├── basic_data_inspection.py
│       ├── bivariate_analysis.py
│       ├── missing_values_analysis.py
│       ├── multivariate_analysis.py
│       └── univariate_analysis.py
├── 📦 data/                       # Raw data storage
│   └── archive.zip               # Ames Housing dataset
├── 📂 extracted_data/            # Extracted dataset
│   └── AmesHousing.csv          # Main dataset file
├── 🔧 pipelines/                 # ML pipeline definitions
│   ├── training_pipeline.py     # Training pipeline
│   └── deployment_pipeline.py   # Deployment pipeline
├── ⚙️ steps/                     # Pipeline steps
│   ├── data_ingestion_step.py
│   ├── handle_missing_values_step.py
│   ├── feature_engineering_step.py
│   ├── outlier_detection_step.py
│   ├── data_splitter_step.py
│   ├── model_building_step.py
│   ├── model_evaluator_step.py
│   ├── dynamic_importer.py
│   ├── model_loader.py
│   ├── prediction_service_loader.py
│   └── predictor.py
├── 🛠️ src/                       # Core source code
│   ├── ingest_data.py           # Data ingestion utilities
│   ├── handle_missing_values.py # Missing value handling
│   ├── feature_engineering.py   # Feature engineering
│   ├── outlier_detection.py     # Outlier detection
│   ├── data_splitter.py         # Data splitting utilities
│   ├── model_building.py        # Model training
│   └── model_evaluator.py       # Model evaluation
├── 🐍 venv/                     # Virtual environment
├── 📋 requirements.txt          # Python dependencies
├── 🚀 run_pipeline.py           # Training pipeline runner
├── 🚀 run_deployment.py         # Deployment pipeline runner
├── 🔮 sample_predict.py         # Sample prediction script
└── 📝 NOTE.md                   # Setup instructions
```

## 🔧 Prerequisites

Before running this project, ensure you have:

- **Python 3.8+** installed on your system
- **Git** for cloning the repository
- **Basic understanding** of Python and machine learning concepts
- **Internet connection** for downloading dependencies

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd House_Price_Prediction
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install ZenML and MLflow Integration

```bash
pip install zenml
zenml integration install mlflow -y
```

### 5. Setup ZenML Cloud Workspace

```bash
# Login to your ZenML workspace
zenml login ml_house_price_prediction

# Set your project
zenml project set house_price_prediction
```

### 6. Register MLflow Components

```bash
# Register experiment tracker
zenml experiment-tracker register mlflow_tracker --flavor=mlflow

# Register model deployer
zenml model-deployer register mlflow --flavor=mlflow
```

### 7. Register and Set the Stack

```bash
zenml stack register local-mlflow-stack -a default -o default -d mlflow -e mlflow_tracker --set
```### 8. Verify Setup

```bash
zenml stack describe
```

You should see all four components: ORCHESTRATOR, ARTIFACT_STORE, EXPERIMENT_TRACKER, MODEL_DEPLOYER.

## 🚀 Quick Start

### Option 1: Run Training Pipeline Only

```bash
python run_pipeline.py
```

This will:
- Load and process the Ames Housing dataset
- Handle missing values
- Perform feature engineering
- Detect and remove outliers
- Split data into training and testing sets
- Train a Linear Regression model
- Evaluate the model performance
- Start MLflow UI for experiment tracking

### Option 2: Run Complete Deployment Pipeline

```bash
python run_deployment.py
```

This will:
- Run the complete training pipeline
- Deploy the trained model as a REST API service
- Start the prediction service locally

## 📖 Usage

### Training the Model

1. **Run the training pipeline:**
   ```bash
   python run_pipeline.py
   ```

2. **View experiment results:**
   ```bash
   mlflow ui --backend-store-uri <tracking_uri>
   ```

### Making Predictions

1. **Start the prediction service:**
   ```bash
   python run_deployment.py
   ```

2. **Make predictions using the sample script:**
   ```bash
   python sample_predict.py
   ```

3. **Or make API calls directly:**
   ```python
   import requests
   import json
   
   url = "http://127.0.0.1:8000/invocations"
   
   # Sample house data
   data = {
       "dataframe_records": [{
           "Gr Liv Area": 1710.0,
           "Overall Qual": 5,
           "Year Built": 1961,
           # ... other features
       }]
   }
   
   response = requests.post(url, json=data)
   prediction = response.json()
   print(f"Predicted Price: ${prediction['predictions'][0]:,.2f}")
   ```

### Stopping the Service

```bash
python run_deployment.py --stop-service
```

## 🔍 Project Components

### Data Pipeline Steps

1. **Data Ingestion** (`data_ingestion_step.py`)
   - Extracts data from ZIP files
   - Supports multiple file formats
   - Factory pattern for extensibility

2. **Missing Values Handling** (`handle_missing_values_step.py`)
   - Identifies missing values
   - Applies appropriate imputation strategies
   - Handles both numerical and categorical data

3. **Feature Engineering** (`feature_engineering_step.py`)
   - Applies log transformation to skewed features
   - Creates new features
   - Handles feature scaling

4. **Outlier Detection** (`outlier_detection_step.py`)
   - Identifies outliers using IQR method
   - Removes extreme values
   - Preserves data quality

5. **Data Splitting** (`data_splitter_step.py`)
   - Splits data into training and testing sets
   - Maintains data distribution
   - Handles target variable separation

6. **Model Building** (`model_building_step.py`)
   - Trains Linear Regression model
   - Includes preprocessing pipeline
   - Handles categorical encoding
   - Logs experiments with MLflow

7. **Model Evaluation** (`model_evaluator_step.py`)
   - Calculates multiple metrics (MSE, MAE, R²)
   - Generates performance reports
   - Compares with baseline models

### Deployment Components

1. **Continuous Deployment Pipeline**
   - Automatically deploys trained models
   - Manages model versions
   - Handles service updates

2. **Inference Pipeline**
   - Loads deployed models
   - Processes batch predictions
   - Handles real-time requests

## 📊 Model Information

### Algorithm
- **Model Type**: Linear Regression
- **Framework**: scikit-learn
- **Preprocessing**: StandardScaler, OneHotEncoder
- **Feature Engineering**: Log transformation for skewed features

### Performance Metrics
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R-squared (R²)**
- **Root Mean Squared Error (RMSE)**

### Features Used
The model uses various house features including:
- **Physical Characteristics**: Lot area, living area, bedrooms, bathrooms
- **Quality Metrics**: Overall quality, condition
- **Temporal Features**: Year built, year remodeled
- **Location Features**: Neighborhood, zoning
- **Amenities**: Garage, basement, fireplace

## 🌐 API Usage

### Endpoint Information
- **URL**: `http://127.0.0.1:8000/invocations`
- **Method**: POST
- **Content-Type**: application/json

### Request Format
```json
{
    "dataframe_records": [
        {
            "Gr Liv Area": 1710.0,
            "Overall Qual": 5,
            "Year Built": 1961,
            "Lot Area": 9600,
            "Bedroom AbvGr": 3,
            "Full Bath": 1,
            "Half Bath": 0,
            "TotRms AbvGrd": 7,
            "Fireplaces": 2,
            "Garage Cars": 2,
            "Garage Area": 500.0
        }
    ]
}
```

### Response Format
```json
{
    "predictions": [185000.0]
}
```

## 🔧 Troubleshooting

### Common Issues

1. **ZenML Login Issues**
   ```bash
   # Clear ZenML cache
   zenml clean
   # Re-login
   zenml login ml_house_price_prediction
   ```

2. **Port Already in Use**
   ```bash
   # Stop existing services
   python run_deployment.py --stop-service
   # Or kill process on port 8000
   ```

3. **Missing Dependencies**
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt --force-reinstall
   ```

4. **MLflow UI Not Starting**
   ```bash
   # Check if tracking URI is correct
   zenml stack describe
   # Use the tracking URI from the output
   ```

### Getting Help

- Check the `NOTE.md` file for detailed setup instructions
- Review the MLflow UI for experiment details
- Check ZenML logs for pipeline issues
- Ensure all dependencies are properly installed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ames Housing Dataset**: Used for training and evaluation
- **ZenML**: MLOps platform for pipeline orchestration
- **MLflow**: Experiment tracking and model deployment
- **scikit-learn**: Machine learning algorithms and utilities
- **pandas**: Data manipulation and analysis

---

**Note**: This project is designed for educational and demonstration purposes. For production use, additional considerations such as security, scalability, and monitoring should be implemented. 


