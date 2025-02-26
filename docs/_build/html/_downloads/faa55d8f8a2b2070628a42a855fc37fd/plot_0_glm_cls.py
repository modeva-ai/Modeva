"""
========================================
Logistic Regression (Classification)
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
# Import required modules
from modeva import DataSet
from modeva import TestSuite
from modeva.models import MoLogisticRegression

# %%
# Load and prepare dataset
ds = DataSet()
ds.load(name="TaiwanCredit")
ds.set_random_split()
ds.set_target("FlagDefault")

# %%
# Train model
# ----------------------------------------------------------
model = MoLogisticRegression(name="GLM",
                             feature_names=ds.feature_names,
                             feature_types=ds.feature_types)
model.fit(ds.train_x, ds.train_y)


# Basic accuracy analysis
# ----------------------------------------------------------
ts = TestSuite(ds, model)
results = ts.diagnose_accuracy_table()
results.table

# %%
# Coefficient interpretation
# ----------------------------------------------------------
results = ts.interpret_coef(features=("PAY_1", "PAY_2", "PAY_3", "EDUCATION", "SEX"))
results.plot()

# %%
# Feature importance
# ----------------------------------------------------------
results = ts.interpret_fi()
results.plot()

# %%
# Main effect plot
# ----------------------------------------------------------
results = ts.interpret_effects(features="PAY_1")
results.plot()

# %%
# Local feature importance analysis
# ----------------------------------------------------------
results = ts.interpret_local_fi(dataset="train",
                                sample_index=15,
                                centered=True)
results.plot()

# %%
# Local feature importance with linear coefficients
# ----------------------------------------------------------
results = ts.interpret_local_linear_fi(dataset="test",
                                       sample_index=15,
                                       centered=True)
results.plot()
