"""
========================================
Performance Metrics (Regression)
========================================

Evaluate model performance and residuals.
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
# Import modeva modules
from modeva import DataSet
from modeva import TestSuite
from modeva.models import MoLGBMRegressor
from modeva.models import MoXGBRegressor

# %%
# Load BikeSharing Dataset
ds = DataSet()
ds.load(name="BikeSharing")
ds.set_random_split()
ds.set_target("cnt")

ds.scale_numerical(features=("cnt",), method="log1p")
ds.preprocess()

# %%
# Fit a XGBoost model
model1 = MoXGBRegressor()
model1.fit(ds.train_x, ds.train_y)

# %%
# Fit a LGBM model
model2 = MoLGBMRegressor(max_depth=2, verbose=-1, random_state=0)
model2.fit(ds.train_x, ds.train_y.ravel())

# %%
# Basic accuracy analysis
# ----------------------------------------------------------
ts = TestSuite(ds, model1)
results = ts.diagnose_accuracy_table(train_dataset="train", test_dataset="test",
                                     metric=("MAE", "MSE", "R2"))
results.table

# %%
# Compare the XGBoost model with LGBM model
# ----------------------------------------------------------
tsc = TestSuite(ds, models=[model1, model2])
results = tsc.compare_accuracy_table(train_dataset="train", test_dataset="test",
                                     metric=("MAE", "MSE", "R2"))
results.plot("MAE")
