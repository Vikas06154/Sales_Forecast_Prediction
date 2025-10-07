# ===============================================
# Imports Necessary Libraries
# ===============================================
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import joblib

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn

# Hugging Face Hub interaction
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ===============================================
# Data Loading
# ===============================================
# Define paths for the dataset stored on Hugging Face Hub
Xtrain_path = "hf://datasets/vikas0615/Sales-Forecast-Prediction/Xtrain.csv"
Xtest_path = "hf://datasets/vikas0615/Sales-Forecast-Prediction/Xtest.csv"
ytrain_path = "hf://datasets/vikas0615/Sales-Forecast-Prediction/ytrain.csv"
ytest_path = "hf://datasets/vikas0615/Sales-Forecast-Prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).values.ravel()
ytest = pd.read_csv(ytest_path).values.ravel()

print("✅ Data loaded successfully from Hugging Face Hub.")

# ===============================================
# Feature Definition
# ===============================================
# Identify numerical and categorical features
numerical_features = Xtrain.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = Xtrain.select_dtypes(include=['object']).columns.tolist()

print("🔧 Numerical Features:", numerical_features)
print("🔧 Categorical Features:", categorical_features)

# ===============================================
# Class Weight Calculation
# ===============================================
classes = np.unique(ytrain)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=ytrain)
class_weight_dict = dict(zip(classes, class_weights))
print("⚖️ Calculated Class Weights:", class_weight_dict)

# ===============================================
# Preprocessing Steps
# ===============================================
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# ===============================================
# Model Definition
# ===============================================
xgb_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=class_weight_dict[1] / class_weight_dict[0]  # handle imbalance
)

# Create a pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', xgb_model)])

# ===============================================
# Experimentation Tracking with MLflow
# ===============================================
mlflow.set_experiment("MLOps_XGBoost_Experiment")

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [4, 6],
    'classifier__learning_rate': [0.05, 0.1]
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=3, scoring='f1', verbose=2, n_jobs=-1
)

with mlflow.start_run():
    grid_search.fit(Xtrain, ytrain)
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)

print("🏆 Best Model Parameters:", grid_search.best_params_)

# ===============================================
# Prediction and Evaluation
# ===============================================
best_model = grid_search.best_estimator_

ytrain_pred = best_model.predict(Xtrain)
ytest_pred = best_model.predict(Xtest)

print("\n📊 Training Performance:\n", classification_report(ytrain, ytrain_pred))
print("\n📊 Test Performance:\n", classification_report(ytest, ytest_pred))

# Log metrics to MLflow
with mlflow.start_run():
    mlflow.log_metric("train_f1", float(classification_report(ytrain, ytrain_pred, output_dict=True)['weighted avg']['f1-score']))
    mlflow.log_metric("test_f1", float(classification_report(ytest, ytest_pred, output_dict=True)['weighted avg']['f1-score']))

# ===============================================
# Model Storage
# ===============================================
os.makedirs("superkart_project/models", exist_ok=True)
model_path = "superkart_project/models/xgb_best_model.joblib"
joblib.dump(best_model, model_path)
print(f"💾 Best model saved locally at: {model_path}")

# ===============================================
# Uploading Model to Hugging Face
# ===============================================
model_repo_id = "vikas0615/Sales-Forecast-Model"
api = HfApi(token=os.getenv("HF_TOKEN"))

# Check if repo exists, else create it
try:
    api.repo_info(repo_id=model_repo_id, repo_type="model")
    print(f"📦 Repository '{model_repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"🚀 Repository '{model_repo_id}' not found. Creating...")
    create_repo(repo_id=model_repo_id, repo_type="model", private=False)

# Upload the model file
api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="xgb_best_model.joblib",
    repo_id=model_repo_id,
    repo_type="model"
)

print("✅ Model uploaded successfully to Hugging Face Hub.")
