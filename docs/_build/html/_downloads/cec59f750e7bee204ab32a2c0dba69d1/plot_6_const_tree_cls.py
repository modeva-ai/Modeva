"""
=================================================
Tree Ensemble Models (Classification)
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
from modeva.models import (MoLGBMClassifier,
                           MoXGBClassifier,
                           MoCatBoostClassifier,
                           MoGradientBoostingClassifier,
                           MoRandomForestClassifier)

# %% 
# Load and prepare dataset
ds = DataSet()
ds.load(name="TaiwanCredit")
ds.set_random_split()
ds.set_target("FlagDefault")

# %% 
# Train model
# ----------------------------------------------------------
# You may replace the model by anyone of the following, including `MoGradientBoostingClassifier`, `MoRandomForestClassifier`, `MoXGBClassifier`, `MoCatBoostClassifier`
model = MoLGBMClassifier(max_depth=2, verbose=-1, random_state=0)
model.fit(ds.train_x, ds.train_y.ravel())

# %%
# Basic accuracy analysis
# ----------------------------------------------------------
ts = TestSuite(ds, model)
results = ts.diagnose_accuracy_table()
results.table

# %%
# Feature importance analysis
# ----------------------------------------------------------
results = ts.interpret_fi()
results.plot(n_bars=10)

# %%
# Effect importance analysis
# ----------------------------------------------------------
results = ts.interpret_ei()
results.plot(n_bars=10)

# %%
# Local feature importance analysis
# ----------------------------------------------------------
results = ts.interpret_local_fi(dataset='train', sample_index=1, centered=True)
results.plot(n_bars=10)

# %%
# Local effect importance analysis
# ----------------------------------------------------------
results = ts.interpret_local_ei(dataset='train', sample_index=1)
results.plot(n_bars=10)

# %%
# Main effect plot
# ----------------------------------------------------------

# %%
# For numerical feature
results = ts.interpret_effects(features="PAY_1")
results.plot()

# %%
# Extract the detail information of the effect, e.g., split points and values.
results.value["Details"]

# %%
# Main effect plot for categorical feature

# %%
# For categorical feature
results = ts.interpret_effects(features="EDUCATION")
results.plot()

# %%
# For 2 features
results = ts.interpret_effects(features=("PAY_1", "PAY_2"))
results.plot()