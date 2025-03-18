"""
=================================================
Wrapping Arbitrary Regressor
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
# authenticate(token='eaaa4301-b140-484c-8e93-f9f633c8bacb')

# %%
# Import required modules
import numpy as np
import pandas as pd
from modeva import DataSet
from modeva import TestSuite
from modeva.models import MoRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# %%
# Scripts to build a model
# ----------------------------------------------------------
data = fetch_california_housing()
X, y = data.data, data.target  # Use California housing dataset
train_idx, test_idx = train_test_split(np.arange(data.data.shape[0]),
                                       test_size=0.2, random_state=42)

estimator = LGBMRegressor(verbose=-1)
estimator.fit(X[train_idx], y[train_idx]) 

# %%
# Wrap the data into Modeva
# ----------------------------------------------------------
ds = DataSet()
ds.load_dataframe(pd.concat([pd.DataFrame(data.data, columns=data.feature_names),
                  pd.DataFrame(data.target, columns=[data.target_names[0]])], axis=1))
ds.set_train_idx(train_idx)
ds.set_test_idx(test_idx)

# %%
# Wrap the model into Modeva
# ----------------------------------------------------------
def predict_func(X):
    # X should be numpy array, and output should be of shape (X.shape[0], 2)
    return estimator.predict(X)

model = MoRegressor(name="LGBM-arbitrary",
                    predict_function=predict_func)

# %%
# Create test suite for diagnostics
# ----------------------------------------------------------
ts = TestSuite(ds, model)

# %%
# Permutation feature importance
result = ts.explain_pfi()
result.plot()

# %%
# LIME for local explanation
result = ts.explain_lime(sample_index=0)
result.plot()
