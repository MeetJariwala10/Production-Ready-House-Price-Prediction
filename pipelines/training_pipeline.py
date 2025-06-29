import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from zenml import Model, pipeline, step
from steps.data_ingestion_step import data_ingestion_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.feature_engineering_step import feature_engineering_step
from steps.outlier_detection_step import outlier_detection_step
from steps.data_splitter_step import data_splitter_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step




@pipeline(
    model=Model(
        name="price_predictor"
    )
)

def ml_pipeline():

    # Data Ingestion Step
    raw_data = data_ingestion_step(
        file_path = str(project_root / "data" / "archive.zip")
    )

    # Handling missing values step
    filled_data = handle_missing_values_step(raw_data)

    # Feature engineering step
    engineered_data = feature_engineering_step(
        filled_data, strategy='log', features=["Gr Liv Area", "SalePrice"]
    )

    # Outlier detection step
    clean_data = outlier_detection_step(engineered_data, column_name="SalePrice")

    # Data splitting step
    X_train, X_test, y_train, y_test = data_splitter_step(clean_data, target_column="SalePrice")

    # Model building step
    model = model_building_step(X_train=X_train, y_train=y_train)

    # Model evaluation step
    evaluation_metrics, mse = model_evaluator_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )

if __name__ == "__main__":
    run = ml_pipeline()
