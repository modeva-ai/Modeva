"""
=================================================
Tuning with optuna (Experimental)
=================================================

To run this code, you need to have optuna installed.
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
from modeva.models import MoLGBMClassifier
from modeva.models import ModelTuneOptuna
from scipy.stats import uniform, randint


# %%
# Load Dataset
ds = DataSet()
ds.load(name="SimuCredit")
ds.set_random_split()

# %%
# Run HPO
# ----------------------------------------------------------
param_distributions = {"max_depth": [1, 2, 3],
                       "learning_rate": uniform(0.01, 0.3),
                       "n_estimators": randint(1, 100), 
                      }

model = MoLGBMClassifier(verbose=-1)
hpo = ModelTuneOptuna(dataset=ds, model=model)
result = hpo.run(param_distributions=param_distributions,
                 sampler="tpe", # "grid", "random", "tpe", "gs", "cma-es", "qmc"
                 metric=("AUC", "ACC", "LogLoss"),
                 cv=5)
result.table

# %%
result.plot("parallel", figsize=(8, 6))

# %%
result.plot(("max_depth", "AUC"))

# %%
result.plot(("learning_rate", "AUC"))

# %%
result.plot(("n_estimators", "AUC"))

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
