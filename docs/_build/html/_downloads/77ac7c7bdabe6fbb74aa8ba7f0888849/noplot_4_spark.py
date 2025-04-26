"""
=================================================
Wrapping PySpark Models
=================================================

This example requires full licence, and the program will break if you use the trial licence.
"""

# %%
# Installation

# To install the required package, use the following command:
# !pip install modeva

# %%
# Authentication

# To get authentication, use the following command: (To get full access please replace the token to your own token)
# from modeva.utils.authenticate import authenticate
# authenticate(auth_code='eaaa4301-b140-484c-8e93-f9f633c8bacb')

# %%
# Import required modules
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import monotonically_increasing_id
from sklearn.datasets import load_breast_cancer
from modeva import DataSet
from modeva import TestSuite
from modeva.models.wrappers.api import modeva_arbitrary_classifier

# %%
# Scripts for building a pyspark model
# ----------------------------------------------------------

# Initialize Spark session
spark = SparkSession.builder.appName("PySpark-Wrapper-Example").getOrCreate()

# Load and prepare dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['label'] = data.target

# Convert Pandas DataFrame to Spark DataFrame and add an index column
spark_df = spark.createDataFrame(df)
spark_df = spark_df.withColumn("index", monotonically_increasing_id())

# Assemble features into a vector
assembler = VectorAssembler(inputCols=data.feature_names, outputCol="features")
spark_df = assembler.transform(spark_df).select("features", "label", "index")

# Split the data into train and test sets
train, test = spark_df.randomSplit([0.8, 0.2], seed=1234)

# Extract the index for each split
train_indices = np.array([row["index"] for row in train.select("index").collect()])
test_indices = np.array([row["index"] for row in test.select("index").collect()])

# %%
# Train model
lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train)

# %%
# Wrap the data
# ----------------------------------------------------------
ds = DataSet()
ds.load_dataframe(data=df)
ds.set_train_idx(train_indices)
ds.set_test_idx(test_indices)

# %%
# Wrap the PySpark model into Modeva
# ----------------------------------------------------------
def predict_func(X):
    X_spark = spark.createDataFrame(X, schema=data.feature_names.tolist())
    X_spark = assembler.transform(X_spark)
    predictions = lr_model.transform(X_spark).select("prediction")
    return np.array([int(row.prediction) for row in predictions.collect()])

def predict_proba_func(X):
    X_spark = spark.createDataFrame(X, schema=data.feature_names.tolist())
    X_spark = assembler.transform(X_spark)
    probabilities = lr_model.transform(X_spark).select("probability")
    return np.array([row.probability.toArray() for row in probabilities.collect()])

model = modeva_arbitrary_classifier(
    name="PySpark-LogisticRegression",
    predict_function=predict_func,
    predict_proba_function=predict_proba_func
)

# %%
# Create test suite for diagnostics
# ----------------------------------------------------------
ts = TestSuite(ds, model)

# %%
# Accuracy table
results = ts.diagnose_accuracy_table()
results.table
