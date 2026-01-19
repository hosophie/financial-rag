## overall-pipeline
<img width="1480" height="680" alt="overall-pipeline" src="https://github.com/user-attachments/assets/a69e9929-a45b-44b1-b829-9d8f4cba7c03" />


## Reproducibility Checklist

* **Data Access**: Dataset is curated from public annual reports of 30+ corporations. Evaluation test cases are stored in `test_case-backup.csv`.
* **Core Logic**: `build_vectorstore.py` handles preprocessing and indexing; `rag.py` contains the core RAG pipeline logic.
* **Evaluation & Execution**: The pipeline is managed by `batch_test.py` and executed via `run.sh`. LLM **temperature is set to 0** to ensure deterministic outputs and maximum reproducibility.
* **Environment**: Developed and tested on Ubuntu 22.04, CUDA 12.8, and Python 3.12 (see `environment.yml` for details).
* **Released Artifacts**: Structured logs and experimental results are preserved in the `ouput/` directory.

---

## Workflow and Usage Guide

### 1. Vector Database Construction
Run `build_vectorstore.py` to process the raw documents and generate the local vector database (e.g., the `faiss_db_qwen/` directory). 
* To incrementally add new document chunks to an existing database without a full rebuild, use `add_document.py`.

### 2. Configuration & Experiments
Experiment parameters, such as model paths and retrieval settings, are managed in `config.json`. Modify this file to set up different experimental groups or comparative studies.

### 3. Execution (Batch & Regression)
Use the provided shell script to run the evaluation pipeline:
```bash
bash run.sh

