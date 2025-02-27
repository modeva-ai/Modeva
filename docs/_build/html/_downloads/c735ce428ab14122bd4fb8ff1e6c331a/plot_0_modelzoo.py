"""
=================================================
ModelZoo
=================================================

This example demonstrates how to use ModelZoo to manage multiple models,
train them, and perform various analyses using TestSuite.
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
from lightgbm import LGBMClassifier

from modeva import DataSet
from modeva import ModelZoo
from modeva import TestSuite
from modeva.utils.mlflow import set_mlflow_home, get_mlflow_home
from modeva.models.local_model_zoo import LocalModelZoo
from modeva.models.wrappers.api import modeva_sklearn_classifier

# Import model classes
from modeva.models import (
    MoLogisticRegression, MoDecisionTreeClassifier,
    MoLGBMClassifier, MoXGBClassifier, MoCatBoostClassifier,
    MoRandomForestClassifier, MoGradientBoostingClassifier,
    MoGAMINetClassifier, MoReLUDNNClassifier,
    MoGLMTreeBoostClassifier, MoNeuralTreeClassifier
)

# %%
# Configure MLflow settings
# ----------------------------------------------------------
set_mlflow_home(mlflow_home="~/modeva_mlflow")
mlflow_home = get_mlflow_home()

# %%
# Load and prepare dataset
# ----------------------------------------------------------
ds = DataSet()
ds.load(name="TaiwanCredit")
ds.set_random_split()

# %%
# Initialize ModelZoo
# ----------------------------------------------------------
mz = LocalModelZoo(name="TaiwanCredit-Exp", dataset=ds)
print(f"Experiment name: {mz.experiment_name}")
print(f"Experiment ID: {mz.experiment_id}")

# %%
# Add traditional ML models
# ----------------------------------------------------------
mz.add_model(model=MoLGBMClassifier(name="LGBM2", max_depth=2, verbose=-1))
mz.add_model(model=MoXGBClassifier(name="XGB2", max_depth=2))
mz.add_model(model=MoCatBoostClassifier(name="CatBoost2", max_depth=2, silent=True))
mz.add_model(model=MoRandomForestClassifier(name="RF2", max_depth=2))
mz.add_model(model=MoGradientBoostingClassifier(name="GBDT2", max_depth=2))
mz.add_model(model=MoLogisticRegression(
    name="LR", 
    feature_names=ds.feature_names, 
    feature_types=ds.feature_types
))
mz.add_model(model=MoDecisionTreeClassifier(name="DT", max_depth=8))
mz.add_model(model=MoReLUDNNClassifier(name="ReLUDNN"))

# %%
# Add advanced ML models
# ----------------------------------------------------------
mz.add_model(model=MoNeuralTreeClassifier(
    name="NeuralTree",
    nn_temperature=0.001,
    nn_max_epochs=100,
    verbose=False,
    random_state=0
))

# %%
# Add wrapped scikit-learn model
# ----------------------------------------------------------
wrap_estimator = modeva_sklearn_classifier(
    name="LGBM-wrapped",
    estimator=LGBMClassifier(verbose=-1)
)
mz.add_model(model=wrap_estimator)

# %%
# Train all models and show leaderboard
# ----------------------------------------------------------
mz.train_all()
mz.leaderboard(order_by="test AUC")

# %%
# Model interpretation examples
# ----------------------------------------------------------
# Feature importance analysis
model = mz.get_model("ReLUDNN")
ts = TestSuite(ds, model)
results = ts.interpret_fi()
results.plot()

# %%
# Feature effects analysis for different models
model = mz.get_model("LGBM2")
ts = TestSuite(ds, model)
results = ts.interpret_effects(features="PAY_1")
results.plot()

# %%
# Model registration and loading
# ----------------------------------------------------------
# Register all models
for name in mz.models.keys():
    mz.register(name)

# List registered models
registered_models = mz.list_registered_models()
print("Registered models:", registered_models)

# %%
# Load and verify registered models
# ----------------------------------------------------------
ds_new = DataSet()
ds_new.load(name="TaiwanCredit")
ds_new.set_random_split()
mz_new = ModelZoo(name="TaiwanCredit-Exp", dataset=ds_new)

# Verify predictions from loaded models
for name in mz.models.keys():
    loaded_model = mz_new.load_registered_model(name)
    predictions = loaded_model.predict_proba(ds_new.train_x)
    print(f"Model {name} predictions shape: {predictions.shape}")