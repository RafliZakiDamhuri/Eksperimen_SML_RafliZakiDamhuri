import pandas as pd
import os

def preprocess_data(input_path, output_path):
    # 1. Memuat Dataset
    df = pd.read_csv(input_path)
    
    # 2. Preprocessing: Menangani Missing Values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # 3. Preprocessing: Menghapus kolom yang tidak diperlukan
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    # 4. Preprocessing: Encoding Data Kategorikal
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')
    
    # 5. Menyimpan hasil preprocessing
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Otomatisasi Berhasil! Data disimpan di {output_path}")

if __name__ == "__main__":
    # Jalankan fungsi
    preprocess_data('titanic_raw.csv', 'preprocessing/titanic_preprocessed.csv')