import yaml
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from pyspark.sql import SparkSession

def load_config(path="../config/settings.yaml"):
    """Loads configuration from the YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def spark_confusion_matrix(model, spark_df):
    """
    Replicates the functionality needed from the original notebook to evaluate Spark ML models.
    Takes a Spark ML model and a Spark DataFrame, predicts, converts to Pandas, and calculates 
    the confusion matrix and returns a dataframe for classification_report.
    """
    
    # Predict on the Spark DataFrame
    predictions = model.transform(spark_df)
    
    # Select the relevant columns (label and prediction) and convert to Pandas
    pdf = predictions.selectExpr(
        f"cast({spark_df.columns[-1]} as int) as status", # Assuming last column is label (status)
        "cast(prediction as int) as prediction"
    ).toPandas()

    # Calculate confusion matrix using scikit-learn
    cm = confusion_matrix(pdf['status'], pdf['prediction'])
    
    return cm, pdf

def init_spark_session(app_name="DefaultPySparkApp", master="yarn", log_level="WARN"):
    """Initializes and configures the SparkSession."""
    spark = SparkSession.builder \
        .appName(app_name) \
        .master(master) \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel(log_level)
    return spark
