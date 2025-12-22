import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Setup MLflow Lokal (Agar tersimpan di folder mlruns kamu)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Titanic_Final_Experiment")

def train_advance():
    # 2. Muat Data Titanic kamu
    df = pd.read_csv('titanic_preprocessed.csv')
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="Advance_Hyperparameter_Tuning"):
        # 3. Hyperparameter Tuning (Agar ada banyak parameter yang tercatat)
        param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        # 4. Manual Logging (Parameter & Metrik)
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", acc)

        # 5. Artefak Tambahan (Kriteria Advance: minimal 2 artefak)
        # Artefak 1: Visualisasi Confusion Matrix (Sama seperti punya temanmu)
        plt.figure()
        cm = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig("training_confusion_matrix.png")
        mlflow.log_artifact("training_confusion_matrix.png") # Muncul di list artefak

        # Artefak 2: File Teks Informasi Dataset
        with open("dataset_info.txt", "w") as f:
            f.write(f"Eksperimen Titanic Rafli Zaki\n")
            f.write(f"Jumlah baris data: {len(df)}\n")
            f.write(f"Fitur yang digunakan: {list(X.columns)}")
        mlflow.log_artifact("dataset_info.txt") # Muncul di list artefak

        # 6. SIMPAN MODEL (Ini yang bikin kamu LULUS Kriteria 2)
        # Akan menghasilkan folder 'model' berisi MLmodel, conda.yaml, dll.
        mlflow.sklearn.log_model(best_model, "model")
        
        print(f"Berhasil! Akurasi: {acc}. Sekarang cek MLflow UI kamu!")

if __name__ == "__main__":
    train_advance()