import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. SETUP DAGSHUB (Syarat Advance)
# Ganti dengan username dan nama repo DagsHub kamu sendiri
USER_NAME = "RafliZakiDamhuri" 
REPO_NAME = "Titanic-Project" 
dagshub.init(repo_owner=USER_NAME, repo_name=REPO_NAME, mlflow=True)

def train_advance():
    # 2. LOAD DATA
    df = pd.read_csv('titanic_preprocessed.csv')
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("Titanic_Advance_Experiment")

    # 3. MANUAL LOGGING (Syarat Advance: Alih-alih autolog)
    with mlflow.start_run(run_name="Advance_Hyperparameter_Tuning"):
        # Hyperparameter Tuning (Syarat Skilled/Advance)
        param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        acc = accuracy_score(y_test, best_model.predict(X_test))

        # LOGGING MANUAL (Metriks & Params)
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", acc)

        # 4. DUA ARTEFAK TAMBAHAN (Syarat Advance: Minimal 2 artefak)
        # Artefak 1: Gambar Confusion Matrix
        plt.figure()
        cm = confusion_matrix(y_test, best_model.predict(X_test))
        ConfusionMatrixDisplay(cm).plot()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png") #

        # Artefak 2: File Teks Informasi
        with open("info_eksperimen.txt", "w") as f:
            f.write(f"Model: RandomForest\nAccuracy: {acc}\nUser: {USER_NAME}")
        mlflow.log_artifact("info_eksperimen.txt") #

        # 5. LOG MODEL (Standar MLflow yang diminta reviewer)
        # Menghasilkan folder 'model' berisi MLmodel, conda.yaml, model.pkl
        mlflow.sklearn.log_model(best_model, "model")
        
        print(f"Berhasil! Cek dashboard DagsHub kamu sekarang.")

if __name__ == "__main__":
    train_advance()