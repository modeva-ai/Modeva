"""
========================================
First Example with Modeva
========================================

"""

# %%
# Import modeva modules.
from modeva import DataSet
from modeva import TestSuite
from modeva.models import MoXGBRegressor, MoLGBMRegressor

# %%
# Load BikeSharing Dataset
ds = DataSet()
ds.load(name="BikeSharing")
ds.set_random_split()

# %%
# Fit XGB and LGBM models
model1 = MoXGBRegressor(name="XGB")
model1.fit(ds.train_x, ds.train_y)

model2 = MoLGBMRegressor(name="LGBM-2", max_depth=2, verbose=-1)
model2.fit(ds.train_x, ds.train_y)

# %%
# Model Explainability (PDP for hr)
ts = TestSuite(ds, model1)
results = ts.explain_pdp("hr")
results.plot()

# %%
# Model Explainability (PDP for season)
results = ts.explain_pdp("season")
results.plot()

# %%
# Diagnostics (accuracy)
results = ts.diagnose_accuracy_table()
results.plot()

# %%
# Diagnostics (slicing accuracy)
results = ts.diagnose_slicing_accuracy(features=(("hr", ), ("season", )), method="uniform",
                                       bins=10, metric="MSE")
results.plot()

# %%
# Model comparison (slicing accuracy)
tsc = TestSuite(ds, models=[model1, model2])
results = tsc.compare_slicing_accuracy(features="hr", method="quantile",
                                       bins=10, metric="MSE")
results.plot()
