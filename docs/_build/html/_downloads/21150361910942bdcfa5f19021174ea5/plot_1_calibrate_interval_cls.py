"""
=================================================
Calibrating Binary Classifier Prediction Interval
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
import numpy as np
from matplotlib import pylab as plt
from modeva import DataSet
from modeva.models import MoXGBClassifier

# %%
# Build a model
# ----------------------------------------------------------
ds = DataSet()
ds.load(name="TaiwanCredit")
ds.set_random_split()

model = MoXGBClassifier(max_depth=2)
model.fit(ds.train_x, ds.train_y)

# %%
# Calibrate the model
# ----------------------------------------------------------
model.calibrate_interval(X=ds.test_x, y=ds.test_y, alpha=0.1)

# %%
# Get prediction interval
# ----------------------------------------------------------
model.predict_interval(ds.test_x[:5])

# %%
# Rest calibration when needed
# ----------------------------------------------------------
model.reset_calibrate_interval()
