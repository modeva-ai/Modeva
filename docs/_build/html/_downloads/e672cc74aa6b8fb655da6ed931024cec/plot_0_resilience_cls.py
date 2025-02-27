"""
=================================================
Resilience Analysis (Classification)
=================================================

This example demonstrates how to analyze model resilience and robustness
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
# authenticate(token='eaaa4301-b140-484c-8e93-f9f633c8bacb')

# %%
# Import required modules
from modeva import DataSet
from modeva import TestSuite
from modeva.models import MoLGBMClassifier
from modeva.models import MoXGBClassifier

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
# Basic resilience analysis
# -------------------------------------------------------------------
ts = TestSuite(ds, model1)

# %% 
# Worst-sample resilience analysis
results = ts.diagnose_resilience(method="worst-sample", metric="AUC")
results.plot()

# %% 
# Outer-sample resilience analysis
results = ts.diagnose_resilience(method="outer-sample", metric="AUC")
results.plot()

# %% 
# Hard-sample resilience analysis
results = ts.diagnose_resilience(method="hard-sample", metric="ACC")
results.plot()

# %% 
# Worst-cluster resilience analysis (K-means)
results = ts.diagnose_resilience(n_clusters=5, method="worst-cluster", metric="AUC")
results.plot()

# %%
# Analyze data drift between "worst" and "remaining" samples
data_results = ds.data_drift_test(
    **results.value[0.1]["data_info"],
    distance_metric="PSI",
    psi_method="uniform",
    psi_bins=10
)
data_results.plot("summary")

# %%
# Resilience comparison
# ----------------------------------------------------------
tsc = TestSuite(ds, models=[model1, model2])

# %%
# Compare resilience performance of multiple models
results = tsc.compare_resilience(n_clusters=5, method="worst-cluster", metric="AUC")
results.plot()
