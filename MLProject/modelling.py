import pandas as pd
import mlflow
import mlflow.sklearn # Kita butuh ini untuk 'save_model'
# import joblib # Kita tidak pakai joblib lagi
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import sys

# --- 1. KONEKSI KE DAGSHUB (VIA ENV SECRETS) ---
print("--- Memulai Pengecekan Environment Variables ---")
uri = os.environ.get("MLFLOW_TRACKING_URI")
username = os.environ.get("MLFLOW_TRACKING_USERNAME")
password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
print(f"MLFLOW_TRACKING_URI: {uri}")
print(f"MLFLOW_TRACKING_USERNAME: {username}")
print(f"MLFLOW_TRACKING_PASSWORD: {'***' if password else 'None'}")
if not uri or not username or not password:
    print("\nGAGAL: SECRETS (URI, USERNAME, PASSWORD) TIDAK DITEMUKAN!\n")
    sys.exit(1)
else:
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password
    print("--- Kredensial MLflow (DagsHub) BERHASIL di-set secara eksplisit ---")

# --- 2. SET TRACKING URI ---
try:
    mlflow.set_tracking_uri(uri)
    print(f"Berhasil mengatur MLflow Tracking URI ke: {uri}")
except Exception as e:
    print(f"Error saat mengatur MLflow URI: {e}")
    sys.exit(1)

# --- 3. TENTUKAN PATH DATA ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
TRAIN_PATH = os.path.join(SCRIPT_DIR, 'winequality_preprocessing', 'train_processed.csv')
TEST_PATH = os.path.join(SCRIPT_DIR, 'winequality_preprocessing', 'test_processed.csv')

def load_data(train_path, test_path):
    # ... (Fungsi load_data Anda - tidak perlu diubah) ...
    print("Data berhasil dimuat.")
    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
    except FileNotFoundError:
        print(f"\nERROR: File data tidak ditemukan di {train_path}")
        sys.exit(1)
    X_train = df_train.drop('is_good', axis=1)
    y_train = df_train['is_good']
    X_test = df_test.drop('is_good', axis=1)
    y_test = df_test['is_good']
    return X_train, y_train, X_test, y_test

def train_model_tuning(X_train, y_train, X_test, y_test):
    try:
        mlflow.set_experiment("K3_CI_Wine_Classification") 
    except Exception as e:
        print(f"Gagal set experiment: {e}")
        sys.exit(1)
        
    with mlflow.start_run(run_name="K3_CI_Run_Tuning_vFinal") as run:
        print("Memulai MLflow run (Manual Logging)...")
        
        # ... (Langkah 5, 6, 7: Tuning, Log Parameter, Log Metrik - tidak perlu diubah) ...
        print("Memulai GridSearchCV (Tuning)...")
        param_grid = {'n_estimators': [100, 150], 'max_depth': [10, 20]}
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Tuning selesai. Parameter terbaik: {best_params}")
        print("Mencatat (log) parameter terbaik...")
        mlflow.log_params(best_params) 
        mlflow.log_param("cv_folds", 3)
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
        
        # --- PERBAIKAN FINAL DI SINI (Langkah 8) ---
        # (Sesuai temuan Anda, kita gunakan save_model untuk membuat MLmodel)
        
        # Tentukan nama folder output lokal
        model_local_path = "model_local_output"
        
        print(f"Menyimpan model (format MLmodel) secara lokal ke folder: {model_local_path}")
        
        # 1. Simpan model ke folder lokal di server GitHub Actions
        # Ini akan membuat folder 'model_local_output' berisi 'MLmodel', 'model.pkl', dll.
        mlflow.sklearn.save_model(
            sk_model=best_model,
            path=model_local_path
        )
        print("Model (format MLmodel) BERHASIL disimpan secara lokal.")
        
        # 2. (Opsional, tapi bagus) Upload folder itu sebagai artefak ke DagsHub
        try:
            mlflow.log_artifact(
                local_path=model_local_path,
                artifact_path="model_files" 
            )
            print("Folder model berhasil di-log ke DagsHub (Artifacts).")
        except Exception as e:
            print(f"Peringatan: Gagal log artefak model: {e}")
            pass
        # --- PERBAIKAN SELESAI ---

        print(f"\n--- Selesai Run ID: {run.info.run_id} ---")
        print(f"Cek run ini di DagsHub: {uri}/#/experiments/2/runs/{run.info.run_id}")

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data(TRAIN_PATH, TEST_PATH)
    
    if X_train is not None:
        train_model_tuning(X_train, y_train, X_test, y_test)