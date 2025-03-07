"""
=================================================
Mixture of Expert (MoE) Regression
=================================================
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
from modeva import DataSet
from modeva import TestSuite
from modeva.models import MoMoERegressor, MoXGBRegressor

# %% 
# Load and prepare dataset for regression
ds = DataSet()
ds.load(name="BikeSharing")
ds.set_random_split()
ds.set_target("cnt")

ds.scale_numerical(features=("cnt",), method="log1p")
ds.preprocess()

# %% 
# Train models
# ----------------------------------------------------------
model = MoMoERegressor(max_depth=2)
model.fit(ds.train_x, ds.train_y)

# %%
# Basic accuracy analysis
# ----------------------------------------------------------
ts = TestSuite(ds, model)
results = ts.diagnose_accuracy_table()
results.table



# %%
# Local MOE weights interpretation
# ----------------------------------------------------------
results = ts.interpret_local_moe_weights()
results.plot()

# %%
# Data drift test between cluster "1" with the rest samples
# ----------------------------------------------------------
results = ts.interpret_moe_cluster_analysis()
data_results = ds.data_drift_test(**results.value[2]["data_info"],
                                  distance_metric="PSI",
                                  psi_method="uniform",
                                  psi_bins=10)
data_results.plot("summary")

# %%
# Interpret feature importance
# ----------------------------------------------------------
results = ts.interpret_fi()

# %%
# Expert No. 0
results.plot("0")

# %%
# Expert No. 2
results.plot("2")

# %%
# Interpret effect importance
# ----------------------------------------------------------
results = ts.interpret_ei()

# %%
# Expert No. 0
results.plot("0")

# %%
# Expert No. 2
results.plot("2")

# %%
# Interpret effects
# ----------------------------------------------------------
results = ts.interpret_effects(features="hr")

# %%
# Expert No. 0
results.plot("0")

# %%
# Expert No. 2
results.plot("2")

# %%
# Expert of all clusters
results.plot("all")

# %%
# Local feature importance analysis
# ----------------------------------------------------------
results = ts.interpret_local_fi(dataset='train', sample_index=1)

# %%
# Expert No. 0
results.plot("0")

# %%
# Expert No. 2
results.plot("2")

# %%
# Local effect importance analysis
# ----------------------------------------------------------
results = ts.interpret_local_ei(dataset='train', sample_index=1)

# %%
# Expert No. 0
results.plot("0")

# %%
# Expert No. 2
results.plot("2")
