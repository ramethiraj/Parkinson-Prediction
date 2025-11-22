from src.utils import load_config, spark_confusion_matrix, init_spark_session
import sys
import os

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml import Pipeline

# Use logging instead of print in a production Spark job
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_parkinsons_pipeline():
    """Main function to run the PySpark ETL and ML pipeline."""
    
    # 1. Configuration and Setup
    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    spark = init_spark_session(
        app_name=config['spark']['app_name'],
        master=config['spark']['master'],
        log_level=config['spark']['log_level']
    )
    
    data_path = config['data']['data_path']
    feature_cols = config['data']['feature_columns']
    target_col = config['data']['target_column']
    split_ratio = config['pipeline']['split_ratio']
    
    logger.info(f"Starting pipeline. Reading data from: {data_path}")

    # 2. Data Loading and Initial Cleaning
    # Read the data, infer schema (assuming CSV header exists)
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    
    # Drop the 'name' column (usually excluded from ML) and handle nulls
    df = df.drop('name').dropna()
    
    # Rename target column to 'label' for Spark ML compatibility
    df = df.withColumnRenamed(target_col, 'label')
    
    # 3. Feature Engineering and Preparation
    # a. Vector Assembler: Combine feature columns into a single vector
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_unscaled"
    )
    df_assembled = assembler.transform(df)

    # b. Scaler: Standardize features
    scaler = StandardScaler(
        inputCol="features_unscaled",
        outputCol="features",
        withStd=True, 
        withMean=False
    )
    
    # Fit scaler to training data (before split for simplicity, ideally fit on train only)
    scaler_model = scaler.fit(df_assembled)
    df_scaled = scaler_model.transform(df_assembled).select('label', 'features')
    
    # 4. Train/Test Split
    train_spark, test_spark = df_scaled.randomSplit([split_ratio, 1.0 - split_ratio], seed=42)
    logger.info(f"Data split: Train count={train_spark.count()}, Test count={test_spark.count()}")

    # 5. Model Training and Evaluation (Matching original notebook models)
    
    for model_key in config['pipeline']['models_to_run']:
        logger.info(f"\n--- Running {model_key.upper()} Model ---")
        
        if model_key == "lr":
            model = LogisticRegression(labelCol="label", featuresCol="features")
            model_trained = model.fit(train_spark)
        
        elif model_key == "rf":
            params = config['model_params']['rf']
            model = RandomForestClassifier(labelCol="label", featuresCol="features", **params)
            model_trained = model.fit(train_spark)

        elif model_key == "gbt":
            params = config['model_params']['gbt']
            model = GBTClassifier(labelCol="label", featuresCol="features", **params)
            model_trained = model.fit(train_spark)

        # Evaluation using the custom utility function
        cm, pdf = spark_confusion_matrix(model_trained, test_spark)
        
        # NOTE: The classification_report needs to be calculated on the local Pandas DataFrame (pdf)
        from sklearn.metrics import classification_report
        
        logger.info("Confusion Matrix:\n" + str(cm))
        logger.info("Classification Report:\n" + classification_report(pdf['status'], pdf['prediction']))
        
        # 6. Model Persistence (Example)
        output_path = f"/models/{config['spark']['app_name']}/{model_key}"
        # model_trained.write().overwrite().save(output_path)
        logger.info(f"Trained model saved to (simulated): {output_path}")

    spark.stop()


if __name__ == "__main__":
    run_parkinsons_pipeline()
