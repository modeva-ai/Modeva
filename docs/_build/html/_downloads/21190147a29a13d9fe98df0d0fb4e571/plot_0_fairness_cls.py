"""
============================================
Model Fairness Analysis (Classification)
============================================

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
from modeva import DataSet
from modeva import TestSuite
from modeva.models import MoLGBMClassifier
from modeva.models import MoXGBClassifier
from modeva.data.utils.loading import load_builtin_data
from modeva.testsuite.utils.slicing_utils import get_data_info

# %%
# Load and prepare dataset
data = load_builtin_data("TaiwanCredit").drop(['SEX', 'MARRIAGE', 'AGE'], axis=1)

ds = DataSet()
ds.load_dataframe(data.iloc[:5000])
ds.set_target("FlagDefault")
ds.set_random_split()

protected_data = load_builtin_data("TaiwanCredit")[['SEX', 'MARRIAGE', 'AGE']]
ds.set_protected_data(protected_data.iloc[:5000])
ds.set_raw_extra_data(name="oot", data=data.iloc[5000:])
ds.set_protected_extra_data(name="oot", data=protected_data.iloc[5000:])

# %% 
# Train models
model1 = MoXGBClassifier()
model1.fit(ds.train_x, ds.train_y)

model2 = MoLGBMClassifier(max_depth=2, verbose=-1, random_state=0)
model2.fit(ds.train_x.astype(float), ds.train_y.ravel().astype(float))

# %% 
# Basic fairness analysis
# ----------------------------------------------------------
ts = TestSuite(ds, model1)

# %%
# Config protected and reference groups
group_config = {
    "Gender-Male": {"feature": "SEX", "protected": 2.0, "reference": 1.0},
    "Gender-Female": {"feature": "SEX", "protected": 1.0, "reference": 2.0},
    "MARRIAGE": {"feature": "MARRIAGE", "protected": 2.0, "reference": 1.0},
    "AGE": {"feature": "AGE", "protected": {"lower": 60, "lower_inclusive": True},
            "reference": {"upper": 60, "upper_inclusive": False}}
}

# %%
# Calculate adverse impact ratio (AIR)
results = ts.diagnose_fairness(group_config=group_config,
                               favorable_label=1,
                               metric="AIR",
                               threshold=0.8)
results.plot()

# %%
# Check distribution drift of protected and reference groups (example for the "Gender-Male" group)
data_results = ds.data_drift_test(
    **results.value["Gender-Male"]["data_info"],
    distance_metric="PSI",
    psi_method="uniform",
    psi_bins=10
)
data_results.plot(name="summary")

# %%
# Analyze data drift for single variable
data_results.plot(name=("density", "PAY_1"))

# %%
# Slicing fairness analysis
# ----------------------------------------------------------
# Single feature slicing
results = ts.diagnose_slicing_fairness(features="PAY_1",
                                       group_config=group_config,
                                       dataset="test",
                                       metric="AIR")
results.plot()

# %%
# Bivariate features slicing
results = ts.diagnose_slicing_fairness(features=("PAY_1", "BILL_AMT1"),
                                       group_config=group_config,
                                       dataset="test",
                                       metric="AIR",
                                       threshold=0.9)
results.plot(name="Gender-Male")

# %% 
# Batch mode single feature slicing
results = ts.diagnose_slicing_fairness(features=(("BILL_AMT1",), ("BILL_AMT2",), ("BILL_AMT3",)),
                                       group_config=group_config,
                                       dataset="test",
                                       metric="AIR",
                                       method="auto-xgb1", bins=5)
results.table["Gender-Male"]

# %%
# Batch mode 1D Slicing (all features by setting features=None)
results = ts.diagnose_slicing_fairness(features=None,
                                       group_config=group_config,
                                       dataset="test",
                                       metric="AIR",
                                       method="auto-xgb1", bins=5)
results.table["Gender-Male"]

# %%
# Analyze data drift
data_info = get_data_info(res_value=results.value["PAY_1"]["Gender-Male"])
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
# Fairness comparison
# ----------------------------------------------------------
tsc = TestSuite(ds, models=[model1, model2])
results = tsc.compare_fairness(group_config=group_config,
                               metric="AIR",
                               threshold=0.8)
results.plot()

# %%
# Compare robustness performance of multiple models under single slicing feature
result = tsc.compare_slicing_fairness(features="BILL_AMT1",
                                      group_config=group_config,
                                      favorable_label=1,
                                      dataset="test",
                                      metric="AIR")
result.table["Gender-Male"]

# %%
# Unfairness mitigation
# ----------------------------------------------------------
# By adjusting threshold of predict proba
result = ts.diagnose_mitigate_unfair_thresholding(group_config=group_config,
                                                  favorable_label=1,
                                                  dataset="test",
                                                  metric="AIR",
                                                  performance_metric="AUC",
                                                  proba_cutoff=30)
result.plot("Gender-Male", figsize=(8, 5))

# %%
# By binning features
result = ts.diagnose_mitigate_unfair_binning(group_config=group_config,
                                             favorable_label=1,
                                             dataset="test",
                                             metric="AIR",
                                             performance_metric="AUC",
                                             binning_method='uniform',
                                             bins=10)
result.plot("Gender-Male")
