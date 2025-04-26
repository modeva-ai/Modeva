"""
========================================
Overfitting Analysis (Classification)
========================================

This example demonstrates how to analyze model overfitting across different data slices
for classification problems using various slicing methods and metrics.
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
# Import required module
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
model1 = MoXGBClassifier(max_depth=1)
model1.fit(ds.train_x, ds.train_y)

model2 = MoLGBMClassifier(max_depth=2, verbose=-1, random_state=0)
model2.fit(ds.train_x, ds.train_y.ravel())

# %%
# Conduct slicing analysis for overfit regions
# -------------------------------------------------
ts = TestSuite(ds, model1)
results = ts.diagnose_slicing_overfit(
    train_dataset="train", 
    test_dataset="test",
    features="PAY_1", 
    metric="AUC"
)
results.table

# %%
# Visualize the results
results.plot()

# %%
# Analyze data drift between samples above and under the threshold
data_info = get_data_info(res_value=results.value)["PAY_1"]
data_results = ds.data_drift_test(
    **data_info, 
    distance_metric="PSI", 
    psi_method="uniform", 
    psi_bins=10
)
data_results.plot("summary")

# %%
# Single feature density plot
data_results.plot(("density", "PAY_1"))

# %%
# Batch mode 1D slicing analysis
# -------------------------------------------------
results = ts.diagnose_slicing_overfit(
    train_dataset="train", 
    test_dataset="test",
    features=(("PAY_1", ), ("PAY_2",), ("PAY_3", )), 
    method="auto-xgb1", 
    metric="AUC",
    threshold=0.0,
)
results.table


# %%
# Batch mode 1D Slicing (all features by setting features=None)
results = ts.diagnose_slicing_overfit(
    train_dataset="train",
    test_dataset="test",
    features=None,
    method="auto-xgb1",
    metric="AUC",
    threshold=0.0,
)
results.table


# %%
# Analyze data drift for 'PAY_1' feature
data_info = get_data_info(res_value=results.value)["PAY_1"]
data_results = ds.data_drift_test(
    **data_info, 
    distance_metric="PSI", 
    psi_method="uniform", 
    psi_bins=10
)
data_results.plot("summary")

# %%
# 2D feature interaction analysis
# -------------------------------------------------
results = ts.diagnose_slicing_overfit(
    train_dataset="train", 
    test_dataset="test",
    features=("PAY_1", "PAY_2"), 
    method="uniform", 
    metric="AUC",
    threshold=-0.1
)
results.table

# %%
# Analyze data drift for feature interaction
data_info = get_data_info(res_value=results.value)[("PAY_1", "PAY_2")]
data_results = ds.data_drift_test(
    **data_info, 
    distance_metric="PSI", 
    psi_method="uniform", 
    psi_bins=10
)
data_results.plot("summary")

# %%
# Model comparison
# -------------------------------------------------
tsc = TestSuite(ds, models=[model1, model2])
results = tsc.compare_slicing_overfit(
    train_dataset="train", 
    test_dataset="test",
    features="PAY_1", 
    method="quantile", 
    bins=10, 
    metric="AUC"
)
results.table
