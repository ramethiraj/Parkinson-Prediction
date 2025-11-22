# PySpark Parkinson's Prediction Pipeline

## 1. Project Overview 

This repository contains the production code for classifying individuals with Parkinson's disease based on voice recordings, using the **Parkinson's Voice Dataset**. The solution is built on **Apache Spark (PySpark)**, leveraging Spark ML for scalable machine learning.

The analysis, feature engineering, and model selection (Logistic Regression, Random Forest, GBT) were initially performed in the provided Jupyter/Databricks Notebook. This repository refactors that work into a reusable, executable PySpark application.

### Key Technologies
* **Processing:** Apache Spark (PySpark)
* **Models:** Spark ML (Logistic Regression, Random Forest, Gradient-Boosted Trees)
* **Language:** Python

## 2. Setup and Installation 🛠️

### Prerequisites
1.  **Spark Cluster:** Access to a running Apache Spark cluster (or local Spark installation).
2.  **Python 3.x**
3.  **Required Libraries:** Defined in `requirements.txt`.

### Step-by-Step Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/yourusername/pyspark-parkinsons-pipeline.git](https://github.com/yourusername/pyspark-parkinsons-pipeline.git)
    cd pyspark-parkinsons-pipeline
    ```

2.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Data Placement:**
    * The raw dataset (e.g., `parkinsons.data`) must be accessible to the Spark cluster (e.g., on HDFS, S3, or MapR-FS). Update the `data_path` in `config/settings.yaml` accordingly.

## 3. Execution (Spark Submit) 🚀

The pipeline is executed using the standard `spark-submit` command.

1.  **Package the Project:** Zip the `src/` directory to include all modules (`pyspark_pipeline.py`, `utils.py`) in the Spark execution environment.

    ```bash
    zip -r pipeline_src.zip src/
    ```

2.  **Run the PySpark Job:**

    ```bash
    spark-submit \
      --master yarn \
      --deploy-mode cluster \
      --py-files pipeline_src.zip \
      --conf spark.executor.memory=4g \
      src/pyspark_pipeline.py
    ```
    *(Note: Adjust `--master`, `--deploy-mode`, and resource allocation based on your cluster environment.)*

## 4. Configuration

The file `config/settings.yaml` controls all parameters for the job, including data paths, feature columns, and model hyperparameters. Review this file before execution.

---
