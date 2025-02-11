import mlflow
import os

# Jalankan MLflow Project
mlflow_run = mlflow.projects.run(
    uri="MLproject",
    synchronous=True  # Pastikan eksekusi selesai sebelum lanjut
)

# Ambil run_id
run_id = mlflow_run.run_id
print(f"Run ID: {run_id}")

# Simpan ke Environment GitHub Actions
with open(os.getenv("GITHUB_ENV"), "a") as f:
    f.write(f"RUN_ID={run_id}\n")
