"""
=================================================
Wrapping H2O Models
=================================================

This example requires full licence, and the program will break if you use the trial licence.
"""

# %%
# Installation

# To install the required package, use the following command:
# pip install modeva

# %%
# Authentication

# To get authentication, use the following command: (To get full access please replace the token to your own token)
# from modeva.utils.authenticate import authenticate
# authenticate(token='eaaa4301-b140-484c-8e93-f9f633c8bacb')

# %%
# Import required modules
import h2o
from h2o.estimators import H2OGradientBoostingEstimator
from modeva import DataSet
from modeva import TestSuite
from modeva.models.wrappers.api import modeva_arbitrary_classifier

# %%
# Scripts for building a H2O model
# ----------------------------------------------------------
# Initialize H2O
try:
    h2o.shutdown()
except:
    pass
h2o.init()
h2o.no_progress()

# %%
# Load a sample binary classification dataset
data = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/prostate/prostate.csv")
data["CAPSULE"] = data["CAPSULE"].asfactor()  # Convert target column to factor

# Split the dataset into train and test sets
train, test = data.split_frame(ratios=[0.8], seed=1234)

# Define feature and target columns
X_columns = data.columns[2:-1]  # All columns except the target
y_column = "CAPSULE"           # Target column

# %%
# Train H2O model
h2o_model = H2OGradientBoostingEstimator()
h2o_model.train(x=X_columns, y=y_column, training_frame=train)

# %%
# Wrap the data into Modeva
# ----------------------------------------------------------
ds = DataSet()
ds.load_dataframe(data=data.as_data_frame()[X_columns + [y_column]])
ds.set_train_idx(train["ID"].as_data_frame().values.flatten() - 1)
ds.set_test_idx(test["ID"].as_data_frame().values.flatten() - 1)
ds.set_task_type("Classification")

# %%
# Wrap the model into Modeva
# ----------------------------------------------------------
def predict_func(X):
    X_h2o = h2o.H2OFrame(X)  # Convert input to H2O Frame
    X_h2o.col_names = X_columns
    predictions = h2o_model.predict(X_h2o)["predict"]
    return predictions.as_data_frame(use_multi_thread=True).values.flatten()

def predict_proba_func(X):
    X_h2o = h2o.H2OFrame(X)  # Convert input to H2O Frame
    X_h2o.col_names = X_columns
    probabilities = h2o_model.predict(X_h2o)
    return probabilities.as_data_frame(use_multi_thread=True).values[:, 1:]

model = modeva_arbitrary_classifier(
    name="H2O-BinaryClassifier",
    predict_function=predict_func,
    predict_proba_function=predict_proba_func
)

# %%
# Create test suite for diagnostics
# ----------------------------------------------------------
ts = TestSuite(ds, model)

# %%
# Basic accuracy analysis
results = ts.diagnose_accuracy_table()
results.table
