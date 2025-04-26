"""
========================================
Outlier Detection
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
# Load a simulated Friedman data
from sklearn.datasets import make_friedman1

ds = DataSet()
ds.load("BikeSharing")

# %%
# Outlier detection by CBLOF
# ----------------------------------------------------------
results = ds.detect_outlier_cblof(dataset="main", method="kmeans", threshold=0.9)
results.plot()

# %%
# Outlier detection by Isolation forest
# ----------------------------------------------------------
results = ds.detect_outlier_isolation_forest()
results.plot()

# %%
# Outlier detection by PCA
# ----------------------------------------------------------
results = ds.detect_outlier_pca(dataset="main", method="reconst_error")
outliers_sample_index = results.table['outliers'].index
results.plot()

# %%
# View and use outlier detection results
# ----------------------------------------------------------

# %%
# Outliers table
results.table['outliers']

# %%
# non-outliers table
results.table['non-outliers']

# %%
# Evaluate outlier scores of samples
results.func(results.table['outliers'])

# %%
# Evaluate outlier scores of samples
results.func(results.table['non-outliers'])

# %%
# Apply outlier removal
ds.set_inactive_samples(dataset="main", sample_idx=outliers_sample_index)
ds.x.shape
