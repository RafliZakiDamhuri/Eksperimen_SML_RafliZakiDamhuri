import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Aktifkan Autologging (Syarat Basic)
mlflow.autolog()

def train_basic():
    # Load data (Pastikan file sudah ada di folder)
    df = pd.read_csv('Membangun_model/titanic_preprocessed.csv')
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run(run_name="Basic_Model"):
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        print("Basic Model Training Selesai (Autologging Aktif)")

if __name__ == "__main__":
    train_basic()
