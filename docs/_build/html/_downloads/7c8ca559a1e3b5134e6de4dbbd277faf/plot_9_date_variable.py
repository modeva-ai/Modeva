"""
========================================
Dealing with Date Variables
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
# Load BikeSharing Dataset
import pandas as pd
from modeva import DataSet
from modeva.data.utils.loading import load_builtin_data

data = load_builtin_data("BikeSharing")
data['Date'] = (pd.to_datetime('2011-01-01') + pd.to_timedelta(data.index / 24, unit='D')).date
data.head()

# %%
# Create some missing and special values for demo purpose
data["Date"].iloc[:10] = "SV1"
data["Date"].iloc[10:15] = "SV2"
data["Date"].iloc[5:20] = pd.NA
data.head()

# %%
# Load the data into Modeva DataSet
ds = DataSet()
ds.load_dataframe(data)
ds.set_target("cnt")
ds.set_inactive_features(features=('Date', ))
ds.set_random_split(shuffle=False)

# %% Data Preprocessing
# ----------------------------------------------------------
ds.reset_preprocess()
ds.impute_missing(features="Date", method='constant', fill_value="2011-01-01",
                  add_indicators=True, special_values=["SV1", "SV2"])
# Uncomment the following to convert date into binned integers.
# ds.encode_categorical(features=("date", ), method="ordinal")
# ds.bin_numerical(features=("date", ), bins=5)
ds.preprocess()
ds.to_df()

# %%
# Data summary
# ----------------------------------------------------------
result = ds.summary()
result.table["summary"]

# %%
# Data summary results for numerical variables
result.table["numerical"]

# %%
# Data summary results for categorical variables
result.table["categorical"]

# %%
# Data summary results for mixed numerical and categorical variables
result.table["mixed"]

# %%
# Data summary results for date type variables
result.table["date"]


# %%
# EDA 2D
# ----------------------------------------------------------

# %%
# EDA 2D between Date and a numerical feature
result = ds.eda_2d(feature_x="Date", feature_y="cnt")
result.plot()

# %%
# EDA 3D
# ----------------------------------------------------------
result = ds.eda_3d(feature_x="Date", feature_y="hr", feature_z="cnt", sample_size=1000)
result.plot()
