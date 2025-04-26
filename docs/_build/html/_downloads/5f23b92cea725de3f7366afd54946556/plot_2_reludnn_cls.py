"""
=================================================
MoReLUDNN Classification
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
from modeva.models import MoReLUDNNClassifier

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
model = MoReLUDNNClassifier(max_epochs=100, verbose=True)
model.fit(ds.train_x, ds.train_y)

# %% 
# Basic accuracy analysis
# ----------------------------------------------------------
ts = TestSuite(ds, model)
results = ts.diagnose_accuracy_table()
results.table

# Global feature importance
# ----------------------------------------------------------
results = ts.interpret_fi()
results.plot()

# %%
# LLM summary table
# ----------------------------------------------------------
results = ts.interpret_llm_summary(dataset="train")
results.table

# %%
# LLM parallel coordinate plot
# ----------------------------------------------------------
results = ts.interpret_llm_pc(dataset="train")
results.plot()

# %%
# LLM profile plot against a feature
# ----------------------------------------------------------
results = ts.interpret_llm_profile(feature="PAY_1", dataset="train")
results.plot()

# %%
# Local feature importance analysis
# ----------------------------------------------------------
results = ts.interpret_local_linear_fi(dataset="train", sample_index=15, centered=True)
results.plot()

# %%
# Extract the last hidden layer outputs
# ----------------------------------------------------------
model.predict_last_hidden_layer(ds.train_x)
