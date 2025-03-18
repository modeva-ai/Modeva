"""
==============================================
Wrapping Scored Regressor
==============================================

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
# authenticate(token='eaaa4301-b140-484c-8e93-f9f633c8bacb')

# %%
# Import modeva modules
import numpy as np
import pandas as pd
from modeva import DataSet
from modeva import TestSuite
from modeva.models import MoXGBRegressor
from modeva.models import MoScoredRegressor
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split

# %%
# Build a model and save the prediction
# ----------------------------------------------------------
X, y = make_friedman1(n_samples=10000, n_features=10, noise=0.1, random_state=2024)
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X, y, np.arange((len(X))), test_size=0.2, random_state=42)

model = MoXGBRegressor(max_depth=2)
model.fit(X_train, y_train)
prediction = model.predict(X)

data = pd.DataFrame(np.concatenate([X, y.reshape(-1, 1),
                                    prediction.reshape(-1, 1)], 1),
                    columns=['X' + str(i) for i in range(X.shape[1])] + ['Y', "prediction"])

# %%
# Wrap the data into Modeva
# ----------------------------------------------------------
ds = DataSet(name="scored-test-demo")
ds.load_dataframe(data)
ds.set_train_idx(train_idx=train_indices)
ds.set_test_idx(test_idx=test_indices)
ds.set_target(feature="Y")
ds.set_prediction(feature="prediction")

# %%
# Save and load the data (optional)
ds.register(override=True)
reload_ds = DataSet(name="scored-test-demo")
reload_ds.load_registered_data(name="scored-test-demo")

# %%
# Convert the model into Modeva
# ----------------------------------------------------------
model = MoScoredRegressor(dataset=ds)

# %%
# Create test suite for diagnostics
# ----------------------------------------------------------
# Note that the robustness test is not available for scored model
ts = TestSuite(ds, model)

# %%
# Run accuracy test without the model object
results = ts.diagnose_accuracy_table()
results.table

# %%
# Run residual analysis test without the model object
results = ts.diagnose_residual_analysis(features="X1")
results.table

# %%
# Run reliability test without the model object
results = ts.diagnose_reliability()
results.table

# %%
# Run resilience test without the model object
results = ts.diagnose_resilience()
results.table

# %%
# Run slicing accuracy test without the model object
results = ts.diagnose_slicing_accuracy(features="X1",
                                       dataset="main",
                                       metric="MAE",
                                       threshold=0)
results.table

# %%
# Run slicing overfit test without the model object
results = ts.diagnose_slicing_overfit(features="X1",
                                      train_dataset="train",
                                      test_dataset="test",
                                      metric="MAE")
results.table
