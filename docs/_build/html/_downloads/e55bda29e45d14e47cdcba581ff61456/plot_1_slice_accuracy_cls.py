"""
========================================
Sliced Performance (Classification)
========================================

This example demonstrates how to analyze model performance across different data slices
for classification problems using various slicing methods and metrics.
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
from modeva import DataSet
from modeva import TestSuite
from modeva.models import MoLGBMClassifier
from modeva.models import MoXGBClassifier
from modeva.testsuite.utils.slicing_utils import get_data_info

# %%
# Load and prepare data
ds = DataSet(name="TaiwanCredit")
ds.load("TaiwanCredit")
ds.set_target("FlagDefault")
ds.set_inactive_features(["SEX", "MARRIAGE", "AGE"])
ds.set_random_split()

# %%
# Fit a XGBoost model
model1 = MoXGBClassifier()
model1.fit(ds.train_x, ds.train_y)

# %%
# Fit a LGBM model
model2 = MoLGBMClassifier(max_depth=2, verbose=-1, random_state=0)
model2.fit(ds.train_x, ds.train_y.ravel().astype(float))

# %%
# Basic slice accuracy analysis
# ------------------------------

# %%
# Analyze residual feature importance
ts = TestSuite(ds, model1)

# %%
# Categorical feature slicing
results = ts.diagnose_slicing_accuracy(features="EDUCATION", metric="AUC",
                                       threshold=0.65)
results.table

# %%
# Uniform binning (Numerical feature)
results = ts.diagnose_slicing_accuracy(features=(("LIMIT_BAL", ), ("PAY_1", )),
                                       method="uniform",
                                       bins=10, metric="AUC",
                                       threshold=0.65)
results.plot(figsize=(6, 5))

# %%
# Quantile binning (Numerical feature)
results = ts.diagnose_slicing_accuracy(features="LIMIT_BAL",
                                       method="quantile",
                                       bins=10, metric="AUC",
                                       threshold=0.65)
results.plot(figsize=(6, 5))

# %%
# auto-xgb1 binning (Numerical feature)
results = ts.diagnose_slicing_accuracy(features="LIMIT_BAL",
                                       method="auto-xgb1",
                                       bins=10,
                                       metric="AUC",
                                       threshold=0.75)
results.plot(figsize=(6, 5))

# %%
# Custom binning (Numerical feature)
results = ts.diagnose_slicing_accuracy(features="LIMIT_BAL",
                                       method="precompute",
                                       bins={"LIMIT_BAL": (0.0, 50000, 1000000)},
                                       metric="AUC")
results.table

# %%
# Advanced slicing analysis
# ---------------------------

# %%
# Batch mode 1D Slicing
results = ts.diagnose_slicing_accuracy(features=(("PAY_1", ), ("BILL_AMT1",), ("PAY_AMT1", )),
                                       method="quantile", metric="AUC", threshold=0.6)
results.table

# %%
# Batch mode 1D Slicing (all features by setting features=None)
results = ts.diagnose_slicing_accuracy(features=None,
                                       method="quantile", metric="AUC", threshold=0.6)
results.table

# %%
# Slicing with 2 features
results = ts.diagnose_slicing_accuracy(features=("PAY_1", "PAY_AMT1"),
                                       method="uniform",
                                       bins=10,
                                       metric="AUC",
                                       threshold=0.5)
results.table

# %%
# Test distributional difference between weak samples and the rest
data_info = get_data_info(res_value=results.value)
data_results = ds.data_drift_test(**data_info[("PAY_1", "PAY_AMT1")],
                                  distance_metric="PSI",
                                  psi_method="uniform",
                                  psi_bins=10)
data_results.plot("summary")

# %%
# Get the list of available figure names in the result object
data_results.get_figure_names()

# %%
# Generate a plot in the result object using the figure name
data_results.plot(('density', 'PAY_AMT6'))

# %%
# Model comparison
# ------------------

# %%
# Model Comparison of 1D slicing accuracy (Numerical feature)
tsc = TestSuite(ds, models=[model1, model2])
results = tsc.compare_slicing_accuracy(features="PAY_AMT1", method="quantile",
                                       bins=10, metric="AUC")
results.plot(figsize=(6, 5))

# %%
# Model Comparison of 1D slicing accuracy (Categorical feature)
results = tsc.compare_slicing_accuracy(features="EDUCATION", metric="AUC",
                                       threshold=0.6)
results.plot(figsize=(6, 5))
