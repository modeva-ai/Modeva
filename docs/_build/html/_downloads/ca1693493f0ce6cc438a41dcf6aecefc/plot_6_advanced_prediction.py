"""
==============================================
Data with Model Predictions
==============================================

This example requires full licence, and the program will break if you use the trial licence.
"""

# %%
# Import modeva modules
import numpy as np
from modeva import DataSet
from modeva import TestSuite
from modeva.models import MoXGBRegressor
from modeva.models import MoScoredRegressor

# %%
# Load data
ds = DataSet()
ds.load("BikeSharing")
ds.set_random_split()

# %%
# Fit a XGB model
model = MoXGBRegressor(max_depth=2)
model.fit(ds.train_x, ds.train_y)

# %%
# Get XGB predictions and combine it to original dataframe
data = ds.to_df()
data["prediction"] = model.predict(ds.x)
data

# %%
# Next, we will use this combined data to do model validation
new_ds = DataSet(name="scored-test-demo")
new_ds.load_dataframe(data)
new_ds.set_train_idx(train_idx=np.array(ds.train_idx))
new_ds.set_test_idx(test_idx=np.array(ds.test_idx))
new_ds.set_target(feature="cnt")
new_ds.set_prediction(feature="prediction")
new_ds.register(override=True)

# %%
# Reload the model (optional)
reload_ds = DataSet(name="scored-test-demo")
reload_ds.load_registered_data(name="scored-test-demo")

# %%
# Run tests without the model object, note that the robustness test is not available for scored model
model = MoScoredRegressor(dataset=new_ds)
ts = TestSuite(ds, model)

# %%
# Run accuracy test without the model object
results = ts.diagnose_accuracy_table()
results.table

# %%
# Run residual analysis test without the model object
results = ts.diagnose_residual_analysis(features="hr")
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
results = ts.diagnose_slicing_accuracy(features="hr", dataset="main", metric="MAE", threshold=0)
results.table

# %%
# Run slicing overfit test without the model object
results = ts.diagnose_slicing_overfit(features="hr", train_dataset="train", test_dataset="test", metric="MAE")
results.table
