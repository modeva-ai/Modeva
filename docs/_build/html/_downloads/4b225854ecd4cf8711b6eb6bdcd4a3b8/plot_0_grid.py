"""
=================================================
Grid Search
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
from modeva.models import MoLGBMClassifier
from modeva.models import ModelTuneGridSearch


# %%
# Load Dataset
ds = DataSet()
ds.load(name="SimuCredit")
ds.set_random_split()


# %%
# Run grid search
# ----------------------------------------------------------
param_grid = {"n_estimators": [50, 100, 200],
              "learning_rate": [(i + 1) * 0.01 for i in range(5)]}
model = MoLGBMClassifier(max_depth=2, verbose=-1)
hpo = ModelTuneGridSearch(dataset=ds, model=model)
result = hpo.run(param_grid=param_grid,
                 metric=("AUC", "ACC", "LogLoss", "Brier"),
                 cv=5)
result.table

# %%
result.plot("parallel", figsize=(8, 6))

# %%
result.plot(("n_estimators", "AUC"))

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
