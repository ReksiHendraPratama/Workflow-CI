import pandas as pd
import mlflow
import mlflow.sklearn # Pastikan ini di-import
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import sys # Kita butuh 'sys' untuk keluar jika ada error

# --- 1. KONEKSI KE DAGSHUB (VIA ENV SECRETS - WAJIB K3) ---
# Ini adalah perbaikan untuk 'RestException' Anda.
# Kita tidak pakai dagshub.init() di Kriteria 3.
print("--- Memulai Pengecekan Environment Variables ---")

# Ambil nilainya dari GitHub Secrets (diteruskan oleh main.yml)
uri = os.environ.get("MLFLOW_TRACKING_URI")
username = os.environ.get("MLFLOW_TRACKING_USERNAME")
password = os.environ.get("MLFLOW_TRACKING_PASSWORD")

# Debugging (Mencetak nilai, '***' untuk password)
print(f"MLFLOW_TRACKING_URI: {uri}")
print(f"MLFLOW_TRACKING_USERNAME: {username}")
print(f"MLFLOW_TRACKING_PASSWORD: {'***' if password else 'None'}")

# Validasi Paksa (Penting untuk debugging)
if not uri or not username or not password:
    print("\n" + "="*50)
    print("  GAGAL: SECRETS (URI, USERNAME, PASSWORD) TIDAK DITEMUKAN!")
    print("  Pastikan 'env:' di file YAML (main.yml) sudah benar.")
    print("="*50 + "\n")
    sys.exit(1) # Gagal (exit code 1)
else:
    # Set env var ini SECARA EKSPLISIT agar MLflow bisa membacanya
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password
    print("--- Kredensial MLflow (DagsHub) BERHASIL di-set secara eksplisit ---")

# --- 2. SET TRACKING URI (TANPA DAGSHUB.INIT) ---
try:
    # Kita HANYA set tracking URI. 
    # MLflow akan otomatis menggunakan USERNAME dan PASSWORD dari env var.
    mlflow.set_tracking_uri(uri)
    print(f"Berhasil mengatur MLflow Tracking URI ke: {uri}")
except Exception as e:
    print(f"Error saat mengatur MLflow URI: {e}")
    sys.exit(1)

# --- 3. TENTUKAN PATH DATA ---
# (Menggunakan path relatif yang robust dari lokasi script)
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
TRAIN_PATH = os.path.join(SCRIPT_DIR, 'winequality_preprocessing', 'train_processed.csv')
TEST_PATH = os.path.join(SCRIPT_DIR, 'winequality_preprocessing', 'test_processed.csv')

def load_data(train_path, test_path):
    """Memuat data CSV yang sudah diproses dari folder K1."""
    print(f"Memuat data training dari: {train_path}")
    print(f"Memuat data testing dari: {test_path}")
    
    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
    except FileNotFoundError:
        print(f"\nERROR: File data tidak ditemukan.")
        print(f"Path yang dicari: {train_path}")
        print("Pastikan folder 'winequality_preprocessing/' ada di dalam 'MLProject/'")
        sys.exit(1)
    
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
    
    # --- 4. SET EKSPERIMEN & MULAI MANUAL LOGGING ---
    try:
        # Kita beri nama Eksperimen (jika belum ada)
        mlflow.set_experiment("K3_CI_Wine_Classification") 
    except Exception as e:
        print(f"Gagal set experiment: {e}")
        print("Ini BISA terjadi jika kredensial Anda (Token/URI) 100% salah.")
        sys.exit(1)
        
    with mlflow.start_run(run_name="K3_CI_Run_Tuning") as run:
        print("Memulai MLflow run (Manual Logging)...")
        
        # --- 5. HYPERPARAMETER TUNING (Kriteria Skilled) ---
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

        # --- 6. LOG PARAMETER (Manual - WAJIB 4 Poin) ---
        print("Mencatat (log) parameter terbaik...")
        mlflow.log_params(best_params) 
        mlflow.log_param("cv_folds", 3)

        # --- 7. LOG METRIK (+2 TAMBAHAN - Manual - WAJIB 4 Poin) ---
        y_pred = best_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Mencatat (log) metrik...")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision) # Memenuhi syarat 4 poin
        mlflow.log_metric("recall", recall)     # Memenuhi syarat 4 poin
        mlflow.log_metric("f1_score", f1)
        
        # --- 8. LOG MODEL (PERBAIKAN KRITIS UNTUK 'RestException') ---
        print("Mencatat (log) model terbaik sebagai artefak...")
        #
        # Kita GANTI 'name' (yang mencoba mendaftar/register model)
        # menjadi 'artifact_path' (yang hanya menyimpan/log model)
        # Ini akan memperbaiki error 'unsupported endpoint'.
        #
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="best_tuned_model" # <-- INI PERBAIKANNYA
        )

        print(f"\n--- Selesai Run ID: {run.info.run_id} ---")
        print(f"  Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print("-" * 50)
        print(f"Cek run ini di DagsHub: {uri}/#/experiments/1/runs/{run.info.run_id}") # Memberi link langsung

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data(TRAIN_PATH, TEST_PATH)
    
    if X_train is not None:
        train_model_tuning(X_train, y_train, X_test, y_test)