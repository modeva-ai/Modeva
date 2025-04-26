"""
==============================================
Wrapping Scored Classifier
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
# authenticate(auth_code='eaaa4301-b140-484c-8e93-f9f633c8bacb')

# %%
# Import modeva modules
import numpy as np
import pandas as pd
from modeva import DataSet
from modeva import TestSuite
from modeva.models import MoXGBClassifier
from modeva.models import MoScoredClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# %%
# Build a model and save the prediction
# ----------------------------------------------------------
X, y = make_classification(n_samples=10000, n_features=2,
                           n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X, y, np.arange((len(X))), test_size=0.2, random_state=42)

model1 = MoXGBClassifier(max_depth=1)
model1.fit(X_train, y_train)
proba1 = model1.predict_proba(X)[:, 1]

model2 = MoXGBClassifier(max_depth=2)
model2.fit(X_train, y_train)
proba2 = model2.predict_proba(X)[:, 1]

data = pd.DataFrame(np.concatenate([X, y.reshape(-1, 1),
                                    proba1.reshape(-1, 1), proba2.reshape(-1, 1)], 1),
                    columns=['X' + str(i) for i in range(X.shape[1])] + ['Y', "proba_XGB1", "proba_XGB2"])

# %%
# Wrap the data into Modeva
# ----------------------------------------------------------
ds = DataSet(name="scored-test-demo")
ds.load_dataframe(data)
ds.set_train_idx(train_idx=train_indices)
ds.set_test_idx(test_idx=test_indices)
ds.set_target(feature="Y")
ds.set_inactive_features(("proba_XGB1", "proba_XGB2"))

# %%
# Convert the model into Modeva
# ----------------------------------------------------------
scored_model1 = MoScoredClassifier(dataset=ds, prediction_proba_name="proba_XGB1")
scored_model2 = MoScoredClassifier(dataset=ds, prediction_proba_name="proba_XGB2")

# %%
# Create test suite for diagnostics
# ----------------------------------------------------------
# Note that the robustness test is not available for scored model
ts = TestSuite(ds, scored_model1)

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
                                       metric="AUC",
                                       threshold=0)
results.table

# %%
# Run slicing overfit test without the model object
results = ts.diagnose_slicing_overfit(features="X1",
                                      train_dataset="train",
                                      test_dataset="test",
                                      metric="LogLoss")
results.table


# %%
# Compare two scored models
# ----------------------------------------------------------
tsc = TestSuite(ds, models=[scored_model1, scored_model2])

# %%
# Run accuracy test without the model object
results = tsc.compare_accuracy_table()
results.table

# %%
# Run slicing accuracy test without the model object
results = tsc.compare_slicing_accuracy(features="X1",
                                       dataset="test",
                                       metric="AUC")
results.table
