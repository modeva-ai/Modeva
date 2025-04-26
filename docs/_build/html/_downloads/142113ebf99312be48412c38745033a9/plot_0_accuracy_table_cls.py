"""
========================================
Performance Metrics (Classification)
========================================

Evaluate model performance and residuals.
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
# Import modeva modules
from modeva import DataSet
from modeva import TestSuite
from modeva.models import MoLGBMClassifier
from modeva.models import MoXGBClassifier

# %%
# Load BikeSharing Dataset
ds = DataSet()
ds.load(name="TaiwanCredit")
ds.set_random_split()

# %%
# Fit a XGBoost model
model1 = MoXGBClassifier()
model1.fit(ds.train_x, ds.train_y)

# %%
# Fit a LGBM model
model2 = MoLGBMClassifier(max_depth=2, verbose=-1, random_state=0)
model2.fit(ds.train_x, ds.train_y.ravel())

# %%
# Basic accuracy analysis
# ----------------------------------------------------------
ts = TestSuite(ds, model1)
results = ts.diagnose_accuracy_table(train_dataset="train", test_dataset="test",
                                     metric=("ACC", "AUC", "LogLoss"))
results.table

# %%
# Generate confusion matrix (train)
results.plot(name=("confusion_matrix", "train"))

# %%
# Generate confusion matrix (test)
results.plot(name=("confusion_matrix", "test"))

# %%
# Generate roc auc curve (train)
results.plot(name=("roc_auc", "train"))

# %%
# Generate roc auc curve (test)
results.plot(name=("roc_auc", "test"))


# %%
# Generate precision recall curve (train)
results.plot(name=("precision_recall", "train"))

# %%
# Generate precision recall curve (test)
results.plot(name=("precision_recall", "test"))

# %%
# Compare the XGBoost model with LGBM model
# ----------------------------------------------------------
tsc = TestSuite(ds, models=[model1, model2])
results = tsc.compare_accuracy_table(train_dataset="train", test_dataset="test",
                                     metric=("ACC", "AUC", "LogLoss"))
results.plot("AUC")
