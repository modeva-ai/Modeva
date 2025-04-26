"""
========================================
Data Drift Test
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
import numpy as np
from modeva import DataSet

# %%
# Load TaiwanCredit Dataset
ds = DataSet()
ds.load("TaiwanCredit")
ds.set_random_split()

# %%
# Test data drift between train and test sets
results = ds.data_drift_test(dataset1="train",
                             dataset2="test",
                             distance_metric='PSI')
results.plot("summary")

# %%
# Density difference for variable PAY_1
results.plot(("density", "PAY_1"))

# %%
# Density difference for variable EDUCATION
results.plot(("density", "EDUCATION"))

# %%
# Test data drift between the first 1000 test samples and the second 1000 test samples
results = ds.data_drift_test(dataset1="test",
                             dataset2="test",
                             sample_idx1=np.arange(1000),
                             sample_idx2=np.arange(1000, 2000),
                             name1="test-first-1000",
                             name2="test-second-1000",
                             distance_metric='PSI')
results.plot("summary")

# %%
# Density difference for variable PAY_1
results.plot(("density", "PAY_1"))

# %%
# Density difference for variable EDUCATION
results.plot(("density", "EDUCATION"))
