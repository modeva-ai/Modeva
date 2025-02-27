"""
=================================================
Particle Swarm Optimization Search
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
from modeva.models import MoLGBMClassifier
from modeva.models import ModelTunePSO


# %%
# Load Dataset
ds = DataSet()
ds.load(name="TaiwanCredit")
ds.set_random_split()


# %%
# Run PSO search
# ----------------------------------------------------------
param_bounds = {"max_depth": [1, 4],
                "learning_rate": [0.01, 1.0]}
param_types = {"max_depth": "int"}

model = MoLGBMClassifier(verbose=-1)
hpo = ModelTunePSO(dataset=ds, model=model)
result = hpo.run(param_bounds=param_bounds,
                 param_types=param_types,
                 n_iter=2,
                 n_particles=10,
                 metric=("AUC", "LogLoss"),
                 cv=5)
result.table

# %%
result.plot("parallel", figsize=(8, 6))

# %%
result.plot(("max_depth", "AUC"))

# %%
result.plot(("learning_rate", "AUC"))

# %%
# Retrain model with best hyperparameter
# ----------------------------------------------------------
model_tuned = MoLGBMClassifier(**result.value["params"][0],
                             name="LGBM-Tuned",
                             verbose=-1)
model_tuned.fit(ds.train_x, ds.train_y)
model_tuned


# %%
# Diagnose the tuned model
# ----------------------------------------------------------
ts = TestSuite(ds, model_tuned)
result = ts.diagnose_accuracy_table()
result.table
