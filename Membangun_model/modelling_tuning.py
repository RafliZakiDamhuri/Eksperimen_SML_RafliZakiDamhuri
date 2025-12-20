import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def train_advance():
    # 1. Load Data
    df = pd.read_csv('Membangun_model/titanic_preprocessed.csv')
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Start MLflow Manual Logging
    with mlflow.start_run(run_name="Advance_Tuning_Model"):
        # Hyperparameter Tuning (Syarat Skilled)
        param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Manual Logging Metrics & Params (Syarat Skilled)
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", acc)

        # ARTEFAK 1: Confusion Matrix Plot (Syarat Advance)
        plt.figure(figsize=(6,4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Advance Model')
        plt.savefig("Membangun_model/confusion_matrix.png")
        mlflow.log_artifact("Membangun_model/confusion_matrix.png")

        # ARTEFAK 2: Model Terlatih (Syarat Advance)
        joblib.dump(best_model, "Membangun_model/model_final.pkl")
        mlflow.log_artifact("Membangun_model/model_final.pkl")
        
        print(f"Advance Model Selesai! Accuracy: {acc}")

if __name__ == "__main__":
    train_advance()
