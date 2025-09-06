# Cost_Prediction_MLOPS

## Tech Stack
- **Model**: XGBoost (Scikitlearn)
- **Experiment Tracking**: MLflow  
- **Pipelining**: DVC  
- **Containerization**: Docker  

---

## Steps to Run the Project

### 1. Create Conda Environment
```bash
conda create -n mlops python=3.10 -y
conda activate mlops
```

### 2️. Install Dependencies
```bash
pip install -e .
```

### 3️. Start MLflow Server
```bash
mlflow server     --backend-store-uri sqlite:///mlflow.db     --default-artifact-root ./artifacts     --host 0.0.0.0 --port XXXX
```

### 4️. Update MLflow Tracking URI  
Update the MLflow URI and port in your code.

### 5️. Run Pipeline with DVC
```bash
dvc repro
```

### 6️. Visualize Metrics
Open MLflow dashboard at:
```
http://localhost:XXXX
```

---

## Docker Support
Build Docker image:
```bash
docker build -t cost-prediction .
```

Run container:
```bash
docker run -it cost-prediction
```

---

## ✨ Project Workflow
1. Code & data versioned with **Git + DVC**  
2. Experiments tracked with **MLflow**  
3. Reproducible pipelines with **DVC**  
4. Portable execution using **Docker**  
