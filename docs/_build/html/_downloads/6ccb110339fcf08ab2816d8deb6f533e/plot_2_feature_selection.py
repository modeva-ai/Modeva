"""
========================================
Feature Selection
========================================

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

# %%
# Load data

ds = DataSet()
ds.load("BikeSharing")
ds.set_random_split()

# %%
# Correlation based feature selection
# ----------------------------------------------------------
results = ds.feature_select_corr(threshold=0.2)
results.plot()

# %%
# XGB-PFI based feature selection
# ----------------------------------------------------------
results = ds.feature_select_xgbpfi(threshold=0.01)
results.plot()

# %%
# RCIT based feature selection
# ----------------------------------------------------------
results = ds.feature_select_rcit()
results.plot()

# %%
# Feature selection operations
# ----------------------------------------------------------

# %%
# Set selected features to be active
ds.set_active_features(features=results.value["selected"])
ds.feature_names

# %%
# Conduct another round of feature selection
results = ds.feature_select_xgbpfi(threshold=0.1)
results.plot()

# %%
# Apply another round of feature selection
ds.set_active_features(features=results.value["selected"])
ds.feature_names

# %%
# Revert all feature selection
ds.set_active_features(features=None) # by default, all features are set active
ds.feature_names
