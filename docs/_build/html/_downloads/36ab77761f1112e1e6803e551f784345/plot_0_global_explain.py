"""
========================================
Global Explainability
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
# Import modeva modules
from modeva import DataSet
from modeva import TestSuite
from modeva.models import MoLGBMRegressor

# %%
# Load Dataset
ds = DataSet()
ds.load(name="BikeSharing")
ds.set_random_split()

ds.scale_numerical(features=("cnt",), method="log1p")
ds.preprocess()

# %%
# Train a LGBM model
model = MoLGBMRegressor(verbose=-1)
model.fit(ds.train_x, ds.train_y)

# %%
# Permutation feature importance
# ----------------------------------------------------------
ts = TestSuite(ds, model)
results = ts.explain_pfi(dataset='test', sample_size=2000, n_repeats=5, random_state=0)
results.plot(n_bars=10)

# %%
# H-statistic
# ----------------------------------------------------------
results = ts.explain_hstatistic(features=('hr',
                                          'atemp',
                                          'season',
                                          'holiday',
                                          'hum'),
                                dataset='train', sample_size=2000, percentiles=(0, 1),
                                grid_resolution=10, response_method='auto', random_state=0)
results.table

# %%
# 1D Partial dependency plots
# ----------------------------------------------------------
results = ts.explain_pdp(features="hr", dataset='train', sample_size=2000, percentiles=(0, 1),
                         grid_resolution=10, response_method='auto', random_state=0)
results.plot()

# %%
# 2D Partial dependency plots
# ----------------------------------------------------------
results = ts.explain_pdp(features=("hum", "hr"), dataset="train")
results.plot()

# %%
# 1D ALE
# ----------------------------------------------------------
results = ts.explain_ale(features="hr", dataset='train', sample_size=2000,
                         grid_resolution=10, response_method='auto', random_state=0)
results.plot()

# %%
# 2D ALE
# ----------------------------------------------------------
results = ts.explain_ale(features=("hum", "hr"), dataset="train")
results.plot()
