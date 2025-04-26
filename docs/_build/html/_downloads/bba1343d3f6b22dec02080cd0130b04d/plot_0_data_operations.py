"""
========================================
Basic Dataset Operations
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
# authenticate(auth_code='eaaa4301-b140-484c-8e93-f9f633c8bacb')

# %%
# Import modeva modules
from modeva import DataSet

# %%
# Load the built-in data
# ----------------------------------------------------------
ds = DataSet()
ds.load("TaiwanCredit")
ds

# %%
# Basic data operations
# ----------------------------------------------------------

# %%
# Split data
ds.set_random_split()

# %%
# Set target
ds.set_target("FlagDefault")

# %%
# Set sample weight
ds.set_sample_weight("LIMIT_BAL")

# %%
# Disable features that will not be used for modeling
ds.set_inactive_features(features=('SEX_2.0',
                                   'MARRIAGE_1.0',
                                   'MARRIAGE_2.0'))

# %%
# View feature names
ds.feature_names

# %%
# View feature types
ds.feature_types

# %%
# View training data
ds.train_x, ds.train_y, ds.train_sample_weight

# %%
# View testing data
ds.test_x, ds.test_y, ds.test_sample_weight

# %%
# Register data into MLFlow
# ----------------------------------------------------------
ds.register(override=True)
ds.list_registered_data()

# %%
# Load data from MLFlow
# ----------------------------------------------------------
dsload = DataSet()
dsload.load_registered_data(name="TaiwanCredit")
dsload
