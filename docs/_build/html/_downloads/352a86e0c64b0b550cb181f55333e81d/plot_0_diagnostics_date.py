"""
========================================
Diagnostics Analysis with Date
========================================

Evaluate model with date column.
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
from modeva import DataSet
from modeva import TestSuite
from modeva.models import MoLGBMRegressor

# %%
# Load BikeSharing Dataset
import pandas as pd
from modeva.data.utils.loading import load_builtin_data

data = load_builtin_data("BikeSharing")
data['Date'] = (pd.to_datetime('2011-01-01') + pd.to_timedelta(data.index / 24, unit='D')).date
data.head()

# %%
# Load the data into Modeva DataSet
ds = DataSet()
ds.load_dataframe(data)
ds.set_target("cnt")
ds.set_inactive_features(features=('Date', ))
ds.set_random_split()

# %%
# Fit a LGBM model
model1 = MoLGBMRegressor(name="LGBM1", max_depth=1, n_estimators=20)
model1.fit(ds.train_x, ds.train_y)

model2 = MoLGBMRegressor(name="LGBM2", max_depth=2, n_estimators=20)
model2.fit(ds.train_x, ds.train_y)

# %%
# Visualize the residual against date
# ----------------------------------------------------------
ts = TestSuite(ds, model1)
results = ts.diagnose_residual_analysis(features="Date", dataset="train")
results.plot()

# %%
# Slicing accuracy diagnostics against date
# ----------------------------------------------------------
results = ts.diagnose_slicing_accuracy(features="Date",
                                       method="uniform")
results.plot(figsize=(5, 4))

# %%
# Custom date as split points
dates = pd.to_datetime(["2011-06-30", "2011-12-31", "2012-06-30"])
results = ts.diagnose_slicing_accuracy(features="Date",
                                       method="precompute",
                                       bins= {"Date": dates.tolist()})
results.plot(figsize=(5, 4))

# %%
# 2D slicing with date
results = ts.diagnose_slicing_accuracy(features=("Date", "hr"),
                                       method="uniform")
results.plot()

# %%
# Compare slicing performance with date
tsc = TestSuite(dataset=ds, models=[model1, model1])
results = tsc.compare_slicing_accuracy(features="Date",
                                       method="uniform")
results.plot(figsize=(5, 4))

# %%
# Slicing overfit with date
# ----------------------------------------------------------
results = ts.diagnose_slicing_overfit(features="Date",
                                      method="uniform")
results.plot(figsize=(5, 4))

# %%
# Compare slicing overfit with date
results = tsc.compare_slicing_overfit(features="Date",
                                      method="uniform")
results.plot(figsize=(5, 4))

# %%
# Slicing reliability with date
# ----------------------------------------------------------
results = ts.diagnose_slicing_reliability(features="Date",
                                          method="uniform")
results.plot(figsize=(5, 4))

# %%
# Compare slicing reliability with date
results = tsc.compare_slicing_reliability(features="Date",
                                          method="uniform")
results.plot(figsize=(5, 4))

# %%
# Slicing robustness with date
# ----------------------------------------------------------
results = ts.diagnose_slicing_robustness(features="Date",
                                         method="uniform")
results.plot(figsize=(5, 4))

# %%
# Compare slicing robustness with date
results = tsc.compare_slicing_robustness(features="Date",
                                         method="uniform")
results.plot(figsize=(5, 4))
