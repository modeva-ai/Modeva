"""
============================================
Robustness Analysis (Classification)
============================================

This example demonstrates how to analyze model robustness
for classification problems using various methods and metrics.
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
# Import required modules
from modeva import DataSet
from modeva import TestSuite
from modeva.models import MoLGBMClassifier
from modeva.models import MoXGBClassifier
from modeva.testsuite.utils.slicing_utils import get_data_info

# %% 
# Load and prepare dataset
ds = DataSet()
ds.load(name="TaiwanCredit")
ds.set_random_split()

# %% 
# Train models
model1 = MoXGBClassifier()
model1.fit(ds.train_x, ds.train_y)

model2 = MoLGBMClassifier(max_depth=2, verbose=-1, random_state=0)
model2.fit(ds.train_x, ds.train_y.ravel())

# %% 
# Basic robustness analysis
# ----------------------------------------------------------
ts = TestSuite(ds, model1)
results = ts.diagnose_robustness(perturb_features=("PAY_1", "EDUCATION",),
                                 noise_levels=(0.1, 0.2, 0.3, 0.4),
                                 metric="AUC")
results.table

# %%
# Box plot of robustness performance
results.plot(figsize=(6, 5))

# %%
# Analyze data drift between small and large prediction changes groups
data_results = ds.data_drift_test(**results.value[0.2]["data_info"])
data_results.plot("summary")

# %%
# Slicing robustness analysis
# -------------------------------------------------------------------
# Single feature slicing
results = ts.diagnose_slicing_robustness(features="PAY_1",
                                         perturb_features=("PAY_1", "EDUCATION",),
                                         noise_levels=0.1,
                                         metric="AUC",
                                         method="auto-xgb1",
                                         threshold=0.7)
results.plot()

# %%
# Analyze data drift for a specific feature
data_info = get_data_info(res_value=results.value)["PAY_1"]
data_results = ds.data_drift_test(**data_info,
                                  distance_metric="PSI",
                                  psi_method="uniform",
                                  psi_bins=10)
data_results.plot("summary")

# %%
# Single feature density plot
data_results.plot(("density", "PAY_1"))

# %%
# Bivariate feature slicing
results = ts.diagnose_slicing_robustness(features=("PAY_1", "PAY_2"),
                                         perturb_features=("PAY_1", "EDUCATION",),
                                         noise_levels=0.1,
                                         metric="AUC",
                                         threshold=0.7)
results.table

# %%
# Batch mode single feature slicing
results = ts.diagnose_slicing_robustness(features=(("PAY_1",), ("PAY_2",), ("PAY_3",)),
                                         perturb_features=("PAY_1", "EDUCATION",),
                                         noise_levels=0.1, 
                                         perturb_method="quantile",
                                         metric="AUC",
                                         threshold=0.7)
results.table


# %%
# Batch mode 1D Slicing (all features by setting features=None)
results = ts.diagnose_slicing_robustness(features=None,
                                         perturb_features=("PAY_1", "EDUCATION",),
                                         noise_levels=0.1,
                                         perturb_method="quantile",
                                         metric="AUC",
                                         threshold=0.7)
results.table

# %%
# Analyze data drift for a specific feature
data_info = get_data_info(res_value=results.value)["PAY_1"]
data_results = ds.data_drift_test(**data_info,
                                  distance_metric="PSI",
                                  psi_method="uniform",
                                  psi_bins=10)
data_results.plot("summary")

# %%
# Robustness comparison
# -------------------------------------------------------------------
tsc = TestSuite(ds, models=[model1, model2])

# %%
# Compare resilience performance of multiple models
results = tsc.compare_robustness(perturb_features=("PAY_1", "EDUCATION",),
                                 noise_levels=(0.1, 0.2, 0.3, 0.4),
                                 perturb_method="quantile",
                                 metric="AUC")
results.plot(figsize=(6, 5))

# %%
# Compare robustness performance of multiple models under single slicing feature
results = tsc.compare_slicing_robustness(features="PAY_1", noise_levels=0.1,
                                         method="quantile", metric="AUC")
results.plot()
