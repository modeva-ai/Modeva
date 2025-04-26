"""
========================================
Sliced Performance (Regression)
========================================

This example demonstrates how to analyze model performance across different data slices
for regression problems using various slicing methods and metrics.
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
from modeva import DataSet
from modeva import TestSuite
from modeva.models import MoLGBMRegressor, MoXGBRegressor
from modeva.testsuite.utils.slicing_utils import get_data_info

# %%
# Load and prepare dataset
ds = DataSet()
ds.load(name="BikeSharing")
ds.set_target("cnt")
ds.set_random_split()

ds.scale_numerical(features=("cnt",), method="log1p")
ds.preprocess()

# %%
# Train models
model1 = MoXGBRegressor()
model1.fit(ds.train_x, ds.train_y)

model2 = MoLGBMRegressor(max_depth=2, verbose=-1, random_state=0)
model2.fit(ds.train_x, ds.train_y.ravel())

# %%
# Basic slice accuracy analysis
# ------------------------------

# %%
# Analyze residual feature importance
ts = TestSuite(ds, model1)

# %%
# Categorical feature slicing
results = ts.diagnose_slicing_accuracy(features="season", metric="MAE", threshold=0.2)
results.table

# Uniform binning (Numerical feature)
results = ts.diagnose_slicing_accuracy(features="hr", method="uniform", bins=10, metric="MAE")
results.table

# %%
# Quantile binning (Numerical feature)
results = ts.diagnose_slicing_accuracy(features="hr", method="quantile", bins=10, metric="MAE")
results.table

# %%
# Auto-XGB binning (Numerical feature)
results = ts.diagnose_slicing_accuracy(features="hr", method="auto-xgb1", metric="MAE")
results.table

# %%
# Custom binning (Numerical feature)
results = ts.diagnose_slicing_accuracy(
    features="hr", 
    method="precompute",
    bins={"hr": (0, 5, 10, 20, 23)}, 
    metric="MAE"
)
results.table

# %%
# Advanced slicing analysis
# -------------------------

# Batch mode 1D slicing
results = ts.diagnose_slicing_accuracy(
    features=(("hr", ), ("season",), ("temp", )),
    method="auto-xgb1",
    metric="MAE"
)
results.plot(name="hr", figsize=(6, 6))

# %%
# Batch mode 1D Slicing (all features by setting features=None)
results = ts.diagnose_slicing_accuracy(
    features=None,
    method="auto-xgb1",
    metric="MAE"
)
results.table

# %%
# 2D feature interaction
results = ts.diagnose_slicing_accuracy(
    features=("hr", "season"), 
    method="uniform", 
    bins=10, 
    metric="MAE"
)
results.plot(figsize=(6, 5))

# %%
# Test distributional difference between weak samples and the rest
data_info = get_data_info(res_value=results.value)
data_results = ds.data_drift_test(**data_info[("hr", "season")],
                                  distance_metric="PSI",
                                  psi_method="uniform",
                                  psi_bins=10)
data_results.plot("summary")

# %%
# Get the list of available figure names in the result object
data_results.get_figure_names()

# %%
# Generate a plot in the result object using the figure name
data_results.plot(('density', 'hr'))


# %%
# Model comparison
# ----------------

# Compare models on numerical feature
tsc = TestSuite(ds, models=[model1, model2])
results = tsc.compare_slicing_accuracy(
    features="hr", 
    method="quantile", 
    bins=10, 
    metric="MAE", 
    threshold=0.2
)
results.plot(figsize=(6, 5))

# %%
# Compare models on categorical feature
results = tsc.compare_slicing_accuracy(
    features="season", 
    metric="MAE", 
    threshold=None
)
results.table
