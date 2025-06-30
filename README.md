# 🏠 House Price Prediction Project

A comprehensive Machine Learning project that predicts house prices using the Ames Housing dataset. This project demonstrates a complete ML pipeline from data ingestion to model deployment using modern MLOps practices with ZenML and MLflow.

## 📋 Table of Contents

- [🏠 House Price Prediction Project](#-house-price-prediction-project)
  - [📋 Table of Contents](#-table-of-contents)
  - [🎯 Overview](#-overview)
  - [✨ Features](#-features)
  - [📁 Project Structure](#-project-structure)
  - [🔧 Prerequisites](#-prerequisites)
  - [📦 Installation](#-installation)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Create Virtual Environment](#2-create-virtual-environment)
    - [3. Install Dependencies](#3-install-dependencies)
    - [4. Install ZenML and MLflow Integration](#4-install-zenml-and-mlflow-integration)
    - [5. Setup ZenML Cloud Workspace](#5-setup-zenml-cloud-workspace)
    - [6. Register MLflow Components](#6-register-mlflow-components)
    - [7. Register and Set the Stack](#7-register-and-set-the-stack)

## 🎯 Overview

This project implements an end-to-end machine learning pipeline for house price prediction using the Ames Housing dataset. It showcases modern MLOps practices including:

- **Automated Data Processing**: Handles missing values, feature engineering, and outlier detection
- **Model Training**: Linear Regression model with comprehensive preprocessing
- **Experiment Tracking**: MLflow integration for tracking experiments and model versions
- **Model Deployment**: REST API service for real-time predictions
- **Pipeline Orchestration**: ZenML for managing the entire ML workflow
- 📝 **Documentation**: Detailed analysis and code documentation

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
```