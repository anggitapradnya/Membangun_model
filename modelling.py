import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn


df = pd.read_csv("telco_churn_preprocessing/telco_churn_clean.csv")

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_experiment("Telco_Churn_RF")

mlflow.sklearn.autolog(
    log_models=True,
    log_input_examples=True,
    log_model_signatures=True
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

with mlflow.start_run(run_name="RF_Balanced_200trees"):
    model.fit(X_train, y_train)