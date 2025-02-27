"""
=================================================
Tree Ensemble Models (Regression)
=================================================
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
# Import required modules
from modeva import DataSet
from modeva import TestSuite
from modeva.models import (MoLGBMRegressor,
                           MoXGBRegressor,
                           MoCatBoostRegressor,
                           MoGradientBoostingRegressor,
                           MoRandomForestRegressor)

# %% 
# Load and prepare dataset
ds = DataSet()
ds.load(name="BikeSharing")
ds.set_random_split()
ds.set_target("cnt")

ds.scale_numerical(features=("cnt",), method="log1p")
ds.preprocess()

# %% 
# Train model
# ----------------------------------------------------------
# You may replace the model by anyone of the following, including `MoGradientBoostingRegressor`, `MoRandomForestRegressor`, `MoXGBRegressor`, `MoCatBoostRegressor`
model = MoLGBMRegressor(max_depth=2, verbose=-1, random_state=0)
model.fit(ds.train_x, ds.train_y.ravel())

# %%
# Basic accuracy analysis
# ----------------------------------------------------------
ts = TestSuite(ds, model)
results = ts.diagnose_accuracy_table()
results.table

# %%
# Feature importance analysis
# ----------------------------------------------------------
results = ts.interpret_fi()
results.plot()

# %%
# Effect importance analysis
# ----------------------------------------------------------
results = ts.interpret_ei()
results.plot(n_bars=10)

# %%
# Local feature importance analysis
# ----------------------------------------------------------
results = ts.interpret_local_fi(dataset='train', sample_index=1, centered=True)
results.plot(n_bars=10)

# %%
# Local effect importance analysis
# ----------------------------------------------------------
results = ts.interpret_local_ei(dataset='train', sample_index=1)
results.plot(n_bars=10)

# %%
# Main effect plot
# ----------------------------------------------------------

# %%
# For numerical feature
results = ts.interpret_effects(features="hr")
results.plot()

# %%
# For categorical feature
results = ts.interpret_effects(features="season")
results.plot()