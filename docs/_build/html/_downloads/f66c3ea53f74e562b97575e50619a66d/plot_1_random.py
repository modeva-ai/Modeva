"""
=================================================
Random Search
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
from modeva.models import MoElasticNet
from modeva.models import ModelTuneRandomSearch


# %%
# Load Dataset
ds = DataSet()
ds.load(name="BikeSharing")
ds.set_random_split()

ds.scale_numerical(features=("cnt",), method="log1p")
ds.preprocess()


# %%
# Run random search
# ----------------------------------------------------------
param_grid = {"alpha": [0.1, 1.0, 10],
              "l1_ratio": [(i + 1) * 0.1 for i in range(10)]}

model = MoElasticNet()
hpo = ModelTuneRandomSearch(dataset=ds, model=model)
result = hpo.run(param_distributions=param_grid,
                 n_iter=20,
                 metric="MSE",
                 cv=5)
result.table

# %%
result.plot("parallel", figsize=(8, 6))

# %%
result.plot(("alpha", "MSE"))

# %%
result.plot(("l1_ratio", "MSE"))

# %%
# Retrain model with best hyperparameter
# ----------------------------------------------------------
model_tuned = MoElasticNet(**result.value["params"][0],
                           name="GLM-Tuned")
model_tuned.fit(ds.train_x, ds.train_y)
model_tuned


# %%
# Diagnose the tuned model
# ----------------------------------------------------------
ts = TestSuite(ds, model_tuned)
result = ts.diagnose_accuracy_table()
result.table
