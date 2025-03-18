"""
=================================================
Calibrating Binary Classifier
=================================================

This example requires full licence, and the program will break if you use the trial licence.
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
from copy import deepcopy
from IPython.display import HTML
from modeva import DataSet
from modeva import TestSuite
from modeva.models import MoXGBClassifier
import mocharts as mc

# %%
# Build a model
# ----------------------------------------------------------
ds = DataSet()
ds.load(name="TaiwanCredit")
ds.set_random_split()

model = MoXGBClassifier(name="Raw XGB", max_depth=2)
model.fit(ds.train_x, ds.train_y)

# %%
# Calibrate the model
# ----------------------------------------------------------
model_calibrated = deepcopy(model)
model_calibrated.name = "Calibrated XGB"
model_calibrated.calibrate_proba(X=ds.test_x, y=ds.test_y, method='isotonic')


# %%
# Check proba before and after calibration
# ----------------------------------------------------------
options = mc.scatterplot(model_calibrated.predict_proba(ds.test_x, calibration=False)[:, 1],
                         model_calibrated.predict_proba(ds.test_x, calibration=True)[:, 1])
options.set_xaxis(axis_name="proba before calibration")
options.set_yaxis(axis_name="proba after calibration")
options.figsize = {'width': 500, 'height': 400}

htmlstr = mc.mocharts_plot(options.render(), return_html=True, silent=True)
HTML(htmlstr)


# %%
# Compare the XGBoost model with LGBM model
# ----------------------------------------------------------
tsc = TestSuite(ds, models=[model, model_calibrated])
results = tsc.compare_accuracy_table(train_dataset="train",
                                     test_dataset="test",
                                     metric="LogLoss")
results.table


# %%
# Rest calibration when needed
# ----------------------------------------------------------
model_calibrated.reset_calibrate_proba()
