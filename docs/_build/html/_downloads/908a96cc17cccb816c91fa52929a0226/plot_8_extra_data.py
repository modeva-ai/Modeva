"""
=================================================
Dealing with Extra Data Sets
=================================================

This example requires full licence, and the program will break if you use the trial licence.
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
from modeva.data.utils.loading import load_builtin_data

# %%
# Load BikeSharing dataset as pandas dataframe
data = load_builtin_data("BikeSharing")
data

# %%
# Load the first 5000 rows into Modeva
# ----------------------------------------------------------
ds = DataSet()
ds.load_dataframe(data.iloc[:5000])
ds.set_random_split()
ds.set_inactive_features(features=['yr', 'temp'])
ds.set_target("cnt")

# %%
# Load the samples indexed from 5000 to 8000 as "oot1" data split
# ---------------------------------------------------------------------
ds.set_raw_extra_data(name="oot1", data=data.iloc[5000:8000])
ds.raw_extra_data['oot1']

# %%
# Load the samples indexed from 8000 to 9000 as "oot2" data split
# ---------------------------------------------------------------------
ds.set_raw_extra_data(name="oot2", data=data.iloc[8000:9000])
ds.raw_extra_data['oot2']

# %%
# Load the samples indexed from 9000 to the last one as "oot3" data split
# ---------------------------------------------------------------------------
ds.set_raw_extra_data(name="oot3", data=data.iloc[9000:])
ds.raw_extra_data['oot3']

# %%
# Show the available data splits
# ---------------------------------------------------------------------------
ds.get_data_list()

# %%
# Delete data split (if needed)
# ---------------------------------------------------------------------------
ds.delete_extra_data("oot3")
ds.get_data_list()

# %%
# Get data split by name
# ---------------------------------------------------------------------------
ds.get_data("oot1")
