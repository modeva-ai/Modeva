"""
========================================
Robustness Analysis (Regression)
========================================

This example demonstrates how to analyze model robustness
for regression problems using various methods and metrics.
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
from modeva import DataSet
from modeva import TestSuite
from modeva.models import MoLGBMRegressor
from modeva.models import MoXGBRegressor
from modeva.testsuite.utils.slicing_utils import get_data_info

# %% 
# Load and prepare dataset
ds = DataSet()
ds.load(name="BikeSharing")
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
# Basic robustness analysis
# ----------------------------------------------------------
ts = TestSuite(ds, model1)
results = ts.diagnose_robustness(
    perturb_features=None,
    noise_levels=(0.1, 0.2, 0.3, 0.4),
    metric="MSE"
)
results.plot(figsize=(6, 5))

# %%
# Slicing robustness analysis
# -------------------------------------------------------------------
# Single feature slicing
results = ts.diagnose_slicing_robustness(
    features="hr",
    perturb_features=("hum", "atemp"),
    noise_levels=0.1,
    metric="MAE",
    method="auto-xgb1",
    threshold=0.2
)
results.table

# %%
# Analyze data drift for a specific feature
data_info = get_data_info(res_value=results.value)["hr"]
data_results = ds.data_drift_test(
    **data_info,
    distance_metric="PSI",
    psi_method="uniform",
    psi_bins=10
)
data_results.plot("summary")

# %%
# Bivariate feature slicing
results = ts.diagnose_slicing_robustness(
    features=("hr", "atemp"),
    perturb_features=("hum", "temp"),
    noise_levels=0.1,
    metric="MSE"
)
results.table

# %%
# Batch mode single feature slicing
results = ts.diagnose_slicing_robustness(
    features=(("hr",), ("atemp",), ("season",)),
    perturb_features=("temp", "hum"),
    noise_levels=0.1,
    perturb_method="quantile",
    metric="MSE",
    threshold=0.15
)
results.table

# %%
# Batch mode 1D Slicing (all features by setting features=None)
results = ts.diagnose_slicing_robustness(features=None,
                                         perturb_features=("temp", "hum"),
                                         noise_levels=0.1,
                                         perturb_method="quantile",
                                         metric="MSE",
                                         threshold=0.15
                                         )
results.table

# %%
# Analyze data drift
data_info = get_data_info(res_value=results.value)["hr"]
data_results = ds.data_drift_test(
    **data_info,
    distance_metric="PSI",
    psi_method="uniform",
    psi_bins=10
)
data_results.plot("summary")

# %%
# Single feature density plot
data_results.plot(("density", "hr"))

# %%
# Model robustness comparison
# ----------------------------------------------------------
tsc = TestSuite(ds, models=[model1, model2])

# %%
# Compare resilience performance of multiple models
results = tsc.compare_robustness(
    perturb_features=("hr", "atemp"),
    noise_levels=(0.1, 0.2, 0.3, 0.4),
    perturb_method="quantile",
    metric="MSE"
)
results.plot(figsize=(6, 5))

# %%
# Compare robustness performance of multiple models under single slicing feature
results = tsc.compare_slicing_robustness(
    features="hr",
    noise_levels=0.1,
    method="quantile",
    metric="MSE"
)
results.plot()