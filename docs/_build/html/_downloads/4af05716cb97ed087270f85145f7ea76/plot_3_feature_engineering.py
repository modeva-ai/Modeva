"""
========================================
Data Processing and Feature Engineering
========================================

This example requires full licence, and the program will break if you use the trial licence.
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
import pandas as pd
from modeva import DataSet
from modeva.data.utils.loading import load_builtin_data

# %%
# Manually create data with special and missing values
data = load_builtin_data("TaiwanCredit")
data["LIMIT_BAL"].iloc[:10] = "SV1"
data["PAY_1"].iloc[10:15] = "SV2"
data["EDUCATION"].iloc[5:20] = pd.NA
data["AGE"].iloc[0:5] = pd.NA
data

# %%
# Data load and summary
# ----------------------------------------------------------

# %%
# Load the dataframe into Modeva
ds = DataSet(name="TW-Credit")
ds.load_dataframe(data)
ds.set_random_split()
ds.data.head(20).iloc[:, :10]

# %%
# Check if the data has missing values
results = ds.summary()
results.table["summary"]

# %%
# Check the features with special values.
results.table["mixed"]


# %%
# Reset preprocessing
# ----------------------------
ds.reset_preprocess()


# %%
# Set the data steps
# ----------------------------

# %%
# Impute numerical features, and add an indicator column for missing values
ds.impute_missing(features=ds.feature_names_numerical, method='mean',
                  add_indicators=True)

# Impute categorical features, and add an indicator column for missing values
ds.impute_missing(features=ds.feature_names_categorical, method='most_frequent',
                  add_indicators=True)

# Impute mixed features, and add an indicator column for missing and special values
# The list of special values need to be configured here manually.
ds.impute_missing(features=ds.feature_names_mixed, method='mean',
                  add_indicators=True, special_values=["SV1", "SV2"])

# %%
# Encoding categorical features
ds.encode_categorical(features=("EDUCATION", "SEX"), method="onehot")

# %%
# Encoding categorical features by target encoding. (Note that this will use y, so it's better to use training data.)
ds.encode_categorical(dataset="train", features=("MARRIAGE", ), method="target", target="FlagDefault")

# %%
# Scaling numerical features
ds.scale_numerical(features=("PAY_1", "PAY_2"), method="minmax")
ds.scale_numerical(features=("LIMIT_BAL", ), method="log1p")
ds.scale_numerical(features=("AGE", ), method="square")
ds.scale_numerical(features=("PAY_AMT1", ), method="quantile")
ds.scale_numerical(features=("PAY_1", "PAY_2",), method="log1p")

# %%
# Binning numerical features
ds.bin_numerical(features=("AGE", "PAY_3", ), bins=10)

# %%
# Execute the preprocessing steps defined above
# --------------------------------------------------------------
ds.preprocess()
ds.to_df()
