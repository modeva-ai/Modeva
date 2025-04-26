"""
=================================================
Decision Tree Classification
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
from modeva.models import MoDecisionTreeClassifier

# %%
# Load and prepare dataset
ds = DataSet()
ds.load(name="TaiwanCredit")  # Changed dataset name
ds.set_random_split()
ds.set_target("FlagDefault")

# %%
# Train model
# ----------------------------------------------------------
model = MoDecisionTreeClassifier(max_depth=3)  # Model initialization
model.fit(ds.train_x, ds.train_y)

# %%
# Basic accuracy analysis
# ----------------------------------------------------------
ts = TestSuite(ds, model)
results = ts.diagnose_accuracy_table()
results.table

# %%
# Global tree interpretation
# ----------------------------------------------------------
results = ts.interpret_global_tree()
results.plot()

# %%
# Local tree interpretation
# ----------------------------------------------------------
results = ts.interpret_local_tree(sample_index=0)
results.plot()
