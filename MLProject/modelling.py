import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import sys

# --- PENGECEKAN EKSPLISIT WAJIB ---
print("--- Memulai Pengecekan Environment Variables ---")

# Ambil nilainya
uri = os.environ.get("MLFLOW_TRACKING_URI")
username = os.environ.get("MLFLOW_TRACKING_USERNAME")
password = os.environ.get("MLFLOW_TRACKING_PASSWORD")

# Debugging
print(f"MLFLOW_TRACKING_URI: {uri}")
print(f"MLFLOW_TRACKING_USERNAME: {username}")
print(f"MLFLOW_TRACKING_PASSWORD: {'***' if password else 'None'}")

# Validasi Paksa
if not uri or not username or not password:
    print("\n" + "="*50)
    print("  GAGAL: SECRETS (URI, USERNAME, PASSWORD) TIDAK DITEMUKAN!")
    print("  Pastikan 'env:' di file YAML sudah benar.")
    print("="*50 + "\n")
    sys.exit(1)
else:
    # Set env var ini SECARA EKSPLISIT agar MLflow bisa membacanya
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password
    print("--- Kredensial MLflow (DagsHub) BERHASIL di-set secara eksplisit ---")

# --- 1. KONEKSI KE MLFLOW (TANPA DAGSHUB.INIT) ---
try:
    # Kita HANYA set tracking URI. 
    # MLflow akan otomatis menggunakan USERNAME dan PASSWORD dari env var.
    mlflow.set_tracking_uri(uri)
    print(f"Berhasil mengatur MLflow Tracking URI ke: {uri}")
except Exception as e:
    print(f"Error saat mengatur MLflow URI: {e}")
    sys.exit(1)

# --- 2. TENTUKAN PATH DATA ---
TRAIN_PATH = os.path.join('winequality_preprocessing', 'train_processed.csv')
TEST_PATH = os.path.join('winequality_preprocessing', 'test_processed.csv')

def load_data(train_path, test_path):
    """Memuat data CSV yang sudah diproses dari folder K1."""
    print(f"Memuat data training dari: {train_path}")
    print(f"Memuat data testing dari: {test_path}")
    
    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
    except FileNotFoundError:
        print(f"\nERROR: File data tidak ditemukan.")
        print("Pastikan Anda sudah MENYALIN folder 'winequality_preprocessing/' ke dalam 'Membangun_model/'")
        return None, None, None, None
    
    X_train = df_train.drop('is_good', axis=1)
    y_train = df_train['is_good']
    X_test = df_test.drop('is_good', axis=1)
    y_test = df_test['is_good']
    
    print("Data berhasil dimuat.")
    return X_train, y_train, X_test, y_test

def train_model_tuning(X_train, y_train, X_test, y_test):
    """
    Melatih model dengan TUNING dan MANUAL LOGGING ke DagsHub
    untuk Kriteria 2 (Advance).
    """
    
    # --- 3. SET EKSPERIMEN & MULAI MANUAL LOGGING (WAJIB 4 Poin) ---
    try:
        mlflow.set_experiment("K2_Wine_Classification_Tuning") # Eksperimen terpisah
    except Exception as e:
        print(f"Gagal set experiment: {e}")
        print("Ini BISA terjadi jika kredensial Anda (Token/URI) 100% salah.")
        sys.exit(1)
        
    with mlflow.start_run(run_name="K2_Advance_Run_Tuning") as run:
        print("Memulai MLflow run (Manual Logging)...")
        
        # --- 4. HYPERPARAMETER TUNING (Kriteria Skilled) ---
        param_grid = {
            'n_estimators': [100, 150], 
            'max_depth': [10, 20]
        }
        
        print("Memulai GridSearchCV (Tuning)...")
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Tuning selesai. Parameter terbaik: {best_params}")

        # --- 5. LOG PARAMETER (Manual - WAJIB 4 Poin) ---
        print("Mencatat (log) parameter terbaik...")
        mlflow.log_params(best_params) 
        mlflow.log_param("cv_folds", 3)

        # --- 6. LOG METRIK (+2 TAMBAHAN - Manual - WAJIB 4 Poin) ---
        y_pred = best_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Mencatat (log) metrik...")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # --- 7. LOG MODEL (Manual - WAJIB 4 Poin) ---
        print("Mencatat (log) model terbaik sebagai artefak...")
        mlflow.sklearn.log_model(best_model, "best_tuned_model")

        print(f"\n--- Selesai Run ID: {run.info.run_id} ---")
        print(f"  Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data(TRAIN_PATH, TEST_PATH)
    
    if X_train is not None:
        train_model_tuning(X_train, y_train, X_test, y_test)