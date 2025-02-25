"""
=================================================
Reliability Analysis (Regression)
=================================================

This example demonstrates how to analyze model reliability and calibration
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
ds.set_random_split(random_state=0)

ds.scale_numerical(features=("cnt",), method="log1p")
ds.preprocess()

# %% 
# Train models
model1 = MoXGBRegressor(max_depth=2)
model1.fit(ds.train_x, ds.train_y)

model2 = MoLGBMRegressor(max_depth=2, verbose=-1, random_state=0)
model2.fit(ds.train_x, ds.train_y.ravel().astype(float))

# %% 
# Basic reliability analysis
# ----------------------------------------------------------
ts = TestSuite(ds, model1)

# %%
# As train_dataset == test_dataset, we would split the test data, one for training (calculating the non-conformal scores) and another for evaluation the test_size (0.5) is the proportion of the test data used for training.
results = ts.diagnose_reliability(
    train_dataset="test", 
    test_dataset="test",
    test_size=0.5, 
    alpha=0.1, 
    max_depth=5,
    random_state=0
)
results.table

# %%
# Analyze data drift
data_results = ds.data_drift_test(
    **results.value["data_info"], 
    distance_metric="PSI", 
    psi_method="uniform", 
    psi_bins=10
)

# %%
# Summary PSI of each feature
data_results.plot("summary")

# %%
# Single feature density plot
data_results.plot(("density", "hr"))

# %%
# Slicing reliability
# --------------------------------------------------
# Single feature reliability analysis
results = ts.diagnose_slicing_reliability(
    features="hr", 
    train_dataset="train", 
    test_dataset="test",
    test_size=0.5, 
    metric="coverage", 
    random_state=0
)
results.plot()

# %% 
# Multiple 1D feature reliability analysis
results = ts.diagnose_slicing_reliability(
    features=(("hr",), ("temp",), ("season",)), 
    train_dataset="train", 
    test_dataset="test",
    test_size=0.5, 
    metric="coverage", 
    random_state=0
)
results.plot("hr")

# %%
# Batch mode 1D Slicing (all features by setting features=None)
results = ts.diagnose_slicing_reliability(
    features=None,
    train_dataset="train",
    test_dataset="test",
    test_size=0.5,
    metric="coverage",
    random_state=0
)
results.table

# %% 
# 2D feature interaction reliability analysis
results = ts.diagnose_slicing_reliability(
    features=("hr", "temp"),
    train_dataset="train", 
    test_dataset="test",
    test_size=0.5, 
    random_state=0
)
results.plot()

# %% 
# Model reliability comparison
# -------------------------------------------
tsc = TestSuite(ds, models=[model1, model2])
results = tsc.compare_reliability(
    train_dataset="train", 
    test_dataset="test",
    test_size=0.5, 
    alpha=0.1, 
    max_depth=5,
    random_state=0
)
results.table

# %% 
# Model slicing reliability comparison
results = tsc.compare_slicing_reliability(
    features="hr", 
    train_dataset="train", 
    test_dataset="test",
    test_size=0.5, 
    alpha=0.1, 
    max_depth=5,
    metric="width", 
    random_state=0
)
results.plot()