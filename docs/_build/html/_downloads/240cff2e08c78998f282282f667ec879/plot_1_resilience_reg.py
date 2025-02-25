"""
=========================================================
Resilience Analysis (Regression)
=========================================================

This example demonstrates how to analyze model resilience
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

# %% 
# Load and prepare dataset
ds = DataSet()
ds.load(name="BikeSharing")
ds.set_random_split()

ds.scale_numerical(features=("cnt",), method="log1p")
ds.preprocess()

# %% 
# Train models
model1 = MoXGBRegressor(max_depth=2, random_state=0)
model1.fit(ds.train_x, ds.train_y)

model2 = MoLGBMRegressor(max_depth=2, verbose=-1, random_state=0)
model2.fit(ds.train_x, ds.train_y.ravel())

# %%
# Basic resilience analysis
# -------------------------------------------------------------------
ts = TestSuite(ds, model1)

# %%
# Worst-sample resilience analysis
results = ts.diagnose_resilience(method="worst-sample", metric="MSE")
results.plot()

# %%
# Outer-sample resilience analysis
results = ts.diagnose_resilience(method="outer-sample", metric="MSE")
results.plot()

# %%
# Hard-sample resilience analysis
results = ts.diagnose_resilience(method="hard-sample", metric="MSE")
results.plot()

# %%
# Worst-cluster resilience analysis (K-means)
results = ts.diagnose_resilience(n_clusters=5, method="worst-cluster", metric="MSE")
results.plot()

# %%
# Analyze data drift between "worst" and "remaining" samples
data_results = ds.data_drift_test(
    **results.value[0.1]["data_info"],
    distance_metric="PSI",
    psi_method="uniform",
    psi_bins=10
)
data_results.plot("summary")


# %%
# Resilience comparison
# ----------------------------------------------------------
tsc = TestSuite(ds, models=[model1, model2])

# %%
# Compare resilience performance of multiple models
results = tsc.compare_resilience(n_clusters=5, method="worst-cluster", metric="MSE")
results.plot()
