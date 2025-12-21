import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def train():
    # --- KRITERIA BASIC: Wajib menggunakan autolog ---
    mlflow.autolog()

    # MEMBACA DATASET
    # Gunakan nama file saja karena MLflow berjalan di dalam folder MLProject
    df = pd.read_csv('titanic_preprocessed.csv')
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- KRITERIA ADVANCE: Manual Log minimal 2 Artefak tambahan ---
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, max_depth=5)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # ARTEFAK 1: Confusion Matrix (Gambar)
        plt.figure(figsize=(6,4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Workflow CI')
        plt.savefig("confusion_matrix_ci.png")
        mlflow.log_artifact("confusion_matrix_ci.png")

        # ARTEFAK 2: File Teks Ringkasan Model
        with open("summary.txt", "w") as f:
            f.write(f"Model: RandomForestClassifier\nAccuracy: {acc}\n")
        mlflow.log_artifact("summary.txt")

        print("Model training selesai dalam workflow CI.")

if __name__ == "__main__":
    train()
