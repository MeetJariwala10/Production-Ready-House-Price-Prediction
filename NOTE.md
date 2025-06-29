# ZenML + MLflow Setup Guide (Cloud Workspace & Project)

### Fistly create a workspace and in that create a project and then follow the steps mentioned belo

## 1. Activate Your Virtual Environment
```
.\venv\Scripts\activate
```
(or `source venv/bin/activate` on macOS/Linux)

## 2. Install Requirements
```
pip install -r requirements.txt
```

## 3. Install ZenML and MLflow Integration
```
pip install zenml
zenml integration install mlflow -y
```

## 4. Login to Your ZenML Cloud Workspace
```
zenml login ml_house_price_prediction
```

## 5. Set Your Project
```
zenml project set house_price_prediction
```

## 6. Register MLflow Components (if not already)
```
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
```

## 7. Register and Set the Stack
```
zenml stack register local-mlflow-stack -a default -o default -d mlflow -e mlflow_tracker --set
```

## 8. Verify Your Stack
```
zenml stack describe
```
You should see all four components: ORCHESTRATOR, ARTIFACT_STORE, EXPERIMENT_TRACKER, MODEL_DEPLOYER.

## 9. Run Your Pipeline
```
python pipelines/training_pipeline.py
```

---

**Reference:**
- [ZenML Installation](https://docs.zenml.io/getting-started/installation)
- [ZenML Stacks & Components](https://docs.zenml.io/stacks-and-components/)
- [ZenML MLflow Integration](https://docs.zenml.io/integrations/mlflow)
- [ZenML Cloud Docs](https://docs.zenml.io/cloud/)