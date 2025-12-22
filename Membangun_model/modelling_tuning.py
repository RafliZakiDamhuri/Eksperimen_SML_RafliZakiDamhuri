import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Gunakan folder lokal kamu
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Submission_Rafli_Final")

def train():
    df = pd.read_csv('titanic_preprocessed.csv')
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="Run_Persis_Teman"):
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)

        # ARTEFAK GAMBAR: Biar muncul file .png seperti punya temanmu
        plt.figure()
        cm = confusion_matrix(y_test, model.predict(X_test))
        ConfusionMatrixDisplay(cm).plot()
        plt.savefig("training_confusion_matrix.png")
        mlflow.log_artifact("training_confusion_matrix.png")

        # ARTEFAK MODEL: Ini yang bikin muncul folder 'model' dan file MLmodel
        # JANGAN GANTI NAMA "model" DI BAWAH INI!
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
        
        print("Selesai! Sekarang buka MLflow UI kamu.")

if __name__ == "__main__":
    train()