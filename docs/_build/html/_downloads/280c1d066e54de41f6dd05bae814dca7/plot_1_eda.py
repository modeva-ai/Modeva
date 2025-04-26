"""
========================================
Exploratory Data Analysis
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
# Load TaiwanCredit Dataset
from modeva import DataSet
ds = DataSet()
ds.load("TaiwanCredit")

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
# EDA 1D
# ----------------------------------------------------------

# %%
# EDA 1D by density
result = ds.eda_1d(feature="PAY_1")
result.plot()

# %%
# EDA 1D by histogram
result = ds.eda_1d(feature="BILL_AMT1", plot_type="histogram")
result.plot()

# %%
# EDA 2D
# ----------------------------------------------------------

# %%
# EDA 2D with 2 numerical features
result = ds.eda_2d(feature_x="BILL_AMT1", feature_y="PAY_1", sample_size=1000)
result.plot()

# %%
# EDA 2D with color and smoothing curve
result = ds.eda_2d(feature_x="BILL_AMT1", feature_y="BILL_AMT2", feature_color="SEX", sample_size=1000, 
                   smoother_order=2)
result.plot(figsize=(6, 5))

# %%
# EDA 2D between numerical and categorical variables
result = ds.eda_2d(feature_x="SEX", feature_y="BILL_AMT1")
result.plot()

# %%
# EDA 2D between two categorical and categorical variables
result = ds.eda_2d(feature_x="MARRIAGE", feature_y="SEX")
result.plot()

# %%
# EDA 3D
# ----------------------------------------------------------
result = ds.eda_3d(feature_x="SEX", feature_y="PAY_1", feature_z="BILL_AMT1", feature_color="EDUCATION",
                   sample_size=1000)
result.plot()

# %%
# Correlation
# ----------------------------------------------------------
result = ds.eda_correlation(features=('PAY_1',
                                      'PAY_2',
                                      'PAY_3',
                                      'PAY_4',
                                      'PAY_5',
                                      'PAY_6'),
                            dataset="main", sample_size=10000)
result.plot()

# %%
# PCA
# ----------------------------------------------------------
result = ds.eda_pca(features=("EDUCATION",
                              "MARRIAGE",
                              'PAY_1',
                              'PAY_2',
                              'PAY_3',
                              'PAY_4',
                              'PAY_5',
                              'PAY_6'),
                    n_components=10, dataset="main", sample_size=None)
result.plot()

# %%
# Umap
# ----------------------------------------------------------
result = ds.eda_umap(features=('PAY_1',
 'PAY_2',
 'PAY_3',
 'PAY_4',
 'PAY_5',
 'PAY_6'), n_components=2, dataset="main", sample_size=1000)
result.table
