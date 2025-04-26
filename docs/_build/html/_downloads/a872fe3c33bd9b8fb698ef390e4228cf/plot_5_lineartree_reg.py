"""
=================================================
Linear Tree Regression
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
# authenticate(auth_code='eaaa4301-b140-484c-8e93-f9f633c8bacb')

# %%
# Import required modules
from modeva import DataSet
from modeva import TestSuite
from modeva.models import MoLGBMRegressor, MoGLMTreeBoostRegressor, MoNeuralTreeRegressor

# %%
# Load and prepare dataset
ds = DataSet()
ds.load(name="BikeSharing")
ds.set_random_split()
ds.set_target("cnt")

ds.scale_numerical(method="minmax")
ds.scale_numerical(features=("cnt",), method="log1p")
ds.preprocess()

# %%
# LGBM Linear Tree model
# ----------------------------------------------------------
model = MoLGBMRegressor(linear_trees=True, max_depth=2, verbose=-1, random_state=0)
model.fit(ds.train_x, ds.train_y.ravel())

# %%
# Basic accuracy analysis
ts = TestSuite(ds, model)
results = ts.diagnose_accuracy_table()
results.table

# %% 
# Feature importance analysis
results = ts.interpret_fi()
results.plot()

# %% 
# Local feature importance analysis
results = ts.interpret_local_fi(sample_index=1, centered=True)
results.plot()

# %%
# Main effect plot
results = ts.interpret_effects(features="hr")
results.plot()

# %%
# Boosted GLMTree model
# ----------------------------------------------------------
model = MoGLMTreeBoostRegressor(max_depth=1, n_estimators=100, reg_lambda=0.001,
                                verbose=True, random_state=0)
model.fit(ds.train_x, ds.train_y.ravel())

# %% 
# Basic accuracy analysis
ts = TestSuite(ds, model)
results = ts.diagnose_accuracy_table()
results.table

# %% 
# Main effect plot
results = ts.interpret_effects(features="hr")
results.plot()


# %%
# Neural Tree model with Monotonicity Constraints
# ----------------------------------------------------------
modelnn = MoNeuralTreeRegressor(estimator=model,
                                nn_temperature=0.0001,
                                nn_max_epochs=20, 
                                feature_names=ds.feature_names,
                                mono_increasing_list=("atemp",),
                                mono_decreasing_list=("hum",),
                                mono_sample_size=1000,
                                reg_mono=10,
                                verbose=True,
                                random_state=0)
modelnn.fit(ds.train_x, ds.train_y.ravel())

# %% 
# Basic accuracy analysis
ts = TestSuite(ds, modelnn)
results = ts.diagnose_accuracy_table()
results.table

# %% 
# Feature importance analysis
results = ts.interpret_fi()
results.plot()

# %%
# Main effect plot
results = ts.interpret_effects(features="atemp")
results.plot()
