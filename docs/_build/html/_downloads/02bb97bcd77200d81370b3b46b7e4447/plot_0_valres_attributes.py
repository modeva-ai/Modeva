"""
==============================
ValidationResult - Attributes
==============================

This example demonstrates the basic attributes of the ValidationResults object.
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
# Imports
from modeva import DataSet

# %%
# Load and prepare data
ds = DataSet()
ds.load(name="BikeSharing")


# %%
# Generate results
result = ds.eda_correlation(features=('hr',
                                      'season',
                                      'hum',
                                      'temp',
                                      'atemp',
                                      'holiday'),
                            dataset="main", method="pearson", sample_size=10000)

# %%
# Attributes
# ----------------------
# Note that not all the attributes are used for a single test result.
# In this example, only key, data, table, and options are specified.
print(result.__doc__)

# %%
# Attributes - key
result.key

# %%
# Attributes - data
result.data

# %%
# Attributes - table
result.table

# %%
# Attributes - options
result.options
