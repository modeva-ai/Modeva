"""
========================================
Subsampling
========================================

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

# %%
# Load data
ds = DataSet()
ds.load("BikeSharing")

# %%
# Random subsampling
# ----------------------------------------------------------
ds.set_active_samples()
results = ds.subsample_random(dataset="main", sample_size=1000)
active_samples_index = results.value["sample_idx"]
active_samples_index

# %%
# Apply subsampling by setting active samples
# ----------------------------------------------------------
ds.set_active_samples(dataset="main", sample_idx=active_samples_index)
ds.x.shape

# %%
# Reset subsampling by `ds.set_active_samples()`
# ----------------------------------------------------------
ds.set_active_samples(dataset="main", sample_idx=None)
ds.x.shape
