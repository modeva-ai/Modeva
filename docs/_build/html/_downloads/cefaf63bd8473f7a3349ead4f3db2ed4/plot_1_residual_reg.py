"""
========================================
Residual Analysis (Regression)
========================================

Evaluate model residuals.
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
# Import modeva modules
from modeva import DataSet
from modeva import TestSuite
from modeva.models import MoLGBMRegressor

# %%
# Load BikeSharing Dataset
ds = DataSet()
ds.load(name="BikeSharing")
ds.set_random_split()
ds.set_target("cnt")

ds.scale_numerical(features=("cnt",), method="log1p")
ds.preprocess()

# %%
# Fit a LGBM model
model = MoLGBMRegressor(max_depth=2, verbose=-1, random_state=0)
model.fit(ds.train_x, ds.train_y.ravel())

# %%
# Analyzes residuals feature importance
# ----------------------------------------------------------
ts = TestSuite(ds, model)
results = ts.diagnose_residual_interpret(dataset="train")
results.plot()

# %%
# Visualize the residual against predictor
# ----------------------------------------------------------
results = ts.diagnose_residual_analysis(features="hr", dataset="train")
results.plot()

# %%
# Visualize the residual against response variable
# ----------------------------------------------------------
results = ts.diagnose_residual_analysis(features="cnt", dataset="train")
results.plot()

# %%
# Visualize the residual against model prediction
# ----------------------------------------------------------
results = ts.diagnose_residual_analysis(use_prediction=True, dataset="train")
results.plot()

# %%
# Interpret residual by a XGB depth-2 model
# -------------------------------------------------------------------
results = ts.diagnose_residual_interpret(dataset='test', n_estimators=100, max_depth=2)

# %%
# XGB-2 feature performance
results.plot("feature_importance")

# %%
# XGB-2 effect performance
results.plot("effect_importance")

# %%
# Further interpretation (main effect plot)
ts_residual = results.value["TestSuite"]
ts_residual.interpret_effects("hr", dataset="test").plot()

# %%
# Further interpretation (local interpretation)
ts_residual.interpret_local_fi(sample_index=20).plot()


# %%
# Random forest-based residual clustering analysis (absolute residual)
# -------------------------------------------------------------------
results = ts.diagnose_residual_cluster(
    dataset="test",
    response_type="abs_residual",
    metric="MAE",
    n_clusters=10,
    cluster_method="pam",
    sample_size=2000,
    rf_n_estimators=100,
    rf_max_depth=5,
)
results.table

# %%
# Residual value for each cluster
results.plot("cluster_residual")

# %%
# Performance metric for each cluster
results.plot("cluster_performance")

# %%
# Feature importance of the random forest model
results.plot("feature_importance")

# %%
# Analyze data drift for a specific cluster
data_results = ds.data_drift_test(
    **results.value["clusters"][2]["data_info"],
    distance_metric="PSI",
    psi_method="uniform",
    psi_bins=10
)
data_results.plot("summary")

# %%
data_results.plot(name=('density', 'hr'))

# %%
# Random forest-based residual clustering analysis (perturbed residual)
# ---------------------------------------------------------------------
results = ts.diagnose_residual_cluster(
    dataset="test",
    response_type="abs_residual_perturb",
    metric="MAE",
    n_clusters=10,
    cluster_method="pam",
    sample_size=2000,
    rf_n_estimators=100,
    rf_max_depth=5,
)
results.table


# %%
# Random forest-based residual clustering analysis (prediction interval width)
# ---------------------------------------------------------------------------------
results = ts.diagnose_residual_cluster(
    dataset="test",
    response_type="pi_width",
    metric="MAE",
    n_clusters=10,
    cluster_method="pam",
    sample_size=2000,
    rf_n_estimators=100,
    rf_max_depth=5,
)
results.table


# %%
# Compare residuals cluster of multiple models
# ----------------------------------------------------------
benchmark = MoLGBMRegressor(max_depth=5, verbose=-1, random_state=0)
benchmark.fit(ds.train_x, ds.train_y.ravel())

tsc = TestSuite(ds, models=[model, benchmark])
results = tsc.compare_residual_cluster(dataset="test")
results.table

# %%
results.plot("cluster_performance")

# %%
results.plot("cluster_residual")
