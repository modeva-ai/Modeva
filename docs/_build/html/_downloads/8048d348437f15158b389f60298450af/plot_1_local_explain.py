"""
========================================
Local Explainability
========================================

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
# Import modeva modules
from modeva import DataSet
from modeva import TestSuite
from modeva.models import MoLGBMClassifier

# %%
# Load Dataset
ds = DataSet()
ds.load(name="TaiwanCredit")
ds.set_random_split()

# %%
# Train a LGBM model
model = MoLGBMClassifier(verbose=-1)
model.fit(ds.train_x, ds.train_y)

# %%
# LIME
# ----------------------------------------------------------
ts = TestSuite(ds, model)
results = ts.explain_lime(dataset="test", sample_index=0, centered=False, random_state=0)
results.plot()

# %%
# Baseline-(Kernel) SHAP (a single baseline sample)
# ----------------------------------------------------------
results = ts.explain_shap(dataset="test", sample_index=0,
                          baseline_dataset="train", baseline_sample_index=2024, random_state=0)
results.plot()

# %%
# Baseline-(Kernel) SHAP (a group of baseline samples)
# ----------------------------------------------------------
results = ts.explain_shap(dataset="test", sample_index=0,
                          baseline_dataset="train", baseline_sample_size=200, random_state=0)
results.plot()
