"""
=================================================
GAMINet Classification
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
import numpy as np
import pandas as pd
from modeva import DataSet
from modeva import TestSuite
from modeva.models import MoGAMINetClassifier

# %%
# Load and prepare dataset
ds = DataSet()
ds.load(name="TaiwanCredit")
ds.set_random_split()
ds.set_target("FlagDefault")

ds.scale_numerical(method="minmax")
ds.preprocess()

# %%
# Train model
# ----------------------------------------------------------
model = MoGAMINetClassifier(random_state=0)
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
results.plot()

# %%
# Global effects interpretation
# ----------------------------------------------------------

# %%
# For numerical feature
results = ts.interpret_effects(features="PAY_1")
results.plot()

# %%
# For categorical feature
results = ts.interpret_effects(features="EDUCATION")
results.plot()

# %%
# For 2 features
results = ts.interpret_effects(features=("PAY_1", "PAY_2"))
results.plot()


# %%
# Local feature importance analysis
# ----------------------------------------------------------
results = ts.interpret_local_fi(sample_index=1, centered=True)
results.plot()

# %%
# Another sample in train set
results = ts.interpret_local_ei(dataset='train', sample_index=1)
results.plot()
