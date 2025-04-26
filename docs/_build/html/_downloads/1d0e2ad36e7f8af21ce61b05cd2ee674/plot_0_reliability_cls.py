"""
=================================================
Reliability Analysis (Classification)
=================================================

This example demonstrates how to analyze model reliability and calibration
for classification problems using various methods and metrics.
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
ds.scale_numerical(method="minmax")
ds.preprocess()
ds.set_random_split(random_state=0)

# %%
# Train models
model1 = MoXGBClassifier(max_depth=2)
model1.fit(ds.train_x, ds.train_y)

model2 = MoLGBMClassifier(max_depth=2, verbose=-1, random_state=0)
model2.fit(ds.train_x, ds.train_y.ravel().astype(float))

# %%
# Basic reliability analysis
# ----------------------------------------------------------
ts = TestSuite(ds, model1)

# %%
# As train_dataset == test_dataset, we would split the test data, one for training (calculating the non-conformal scores) and another for evaluation the test_size (0.5) is the proportion of the test data used for training.
results = ts.diagnose_reliability(
    train_dataset="test", 
    test_dataset="test",
    test_size=0.5, 
    alpha=0.2,
    random_state=0
)
results.table

# %%
# Analyze data drift between reliable and unreliable samples of the test dataset (obtained from the reliability analysis)
data_results = ds.data_drift_test(
    **results.value["data_info"], 
    distance_metric="PSI", 
    psi_method="uniform", 
    psi_bins=10
)

# %%
# Draw the PSI values of each feature
data_results.plot("summary")

# %%
# Draw the density plot of the reliable and unreliable samples against "PAY_1"
data_results.plot(("density", "PAY_1"))

# %%
# Slicing reliability
# --------------------------------------------------
# features is the feature to be used for slicing
results = ts.diagnose_slicing_reliability(
    features="PAY_1", 
    train_dataset="train", 
    test_dataset="test",
    test_size=0.5, 
    metric="coverage", 
    random_state=0
)
results.plot()

# %%
# Multiple 1D feature reliability analysis
results = ts.diagnose_slicing_reliability(
    features=(("PAY_1", ), ("EDUCATION",), ("PAY_2", )),
    train_dataset="train", 
    test_dataset="test",
    test_size=0.5, 
    metric="coverage", 
    random_state=0
)
results.table


# %%
# Batch mode 1D Slicing (all features by setting features=None)
results = ts.diagnose_slicing_reliability(
    features=None,
    train_dataset="train",
    test_dataset="test",
    test_size=0.5,
    metric="coverage",
    random_state=0
)
results.table

# %%
# Draw the coverage plot of each feature
results.plot("PAY_1")

# %%
# Analyze data drift between samples above and under the threshold
data_info = get_data_info(res_value=results.value)
data_results = ds.data_drift_test(
    **data_info["PAY_1"],
    distance_metric="PSI",
    psi_method="uniform",
    psi_bins=10
)
data_results.plot("summary")

# %%
# Single feature density plot
data_results.plot(("density", "PAY_1"))


# %%
# 2D feature interaction reliability analysis
# we can use a pair of features for 2D slicing
results = ts.diagnose_slicing_reliability(
    features=("PAY_1", "EDUCATION"), 
    train_dataset="train", 
    test_dataset="test",
    test_size=0.5, 
    random_state=0
)
results.table


# %%
# Model reliability comparison
# -------------------------------------------
tsc = TestSuite(ds, models=[model1, model2])
results = tsc.compare_reliability(
    train_dataset="train", 
    test_dataset="test",
    test_size=0.5,
    alpha=0.1, 
    max_depth=5,
    random_state=0
)
results.table

# %%
# Model slicing reliability comparison
results = tsc.compare_slicing_reliability(
    features="PAY_1",
    train_dataset="train", 
    test_dataset="test",
    test_size=0.5, 
    alpha=0.1, 
    max_depth=5,
    metric="width", 
    random_state=0
)
results.plot()
