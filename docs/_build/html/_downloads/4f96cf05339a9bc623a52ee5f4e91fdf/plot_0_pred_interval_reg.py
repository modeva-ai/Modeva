"""
=================================================
Calibrating Regressor Prediction Interval
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
import mocharts as mc
from IPython.display import HTML
from modeva import DataSet
from modeva.models import MoXGBRegressor

# %%
# Build a model
# ----------------------------------------------------------
ds = DataSet()
ds.load(name="BikeSharing")
ds.set_random_split()
ds.scale_numerical(features=("cnt",), method="log1p")
ds.preprocess()

model = MoXGBRegressor(max_depth=2)
model.fit(ds.train_x, ds.train_y)

# %%
# Calibrate the model
# ----------------------------------------------------------
model.calibrate_interval(X=ds.test_x, y=ds.test_y, alpha=0.1, max_depth=5)

# %%
# Get prediction interval
# ----------------------------------------------------------
print(model.predict_interval(ds.test_x[:5]))

# %%
# Visualize prediction interval
# ----------------------------------------------------------
p = model.predict(ds.test_x)
pi = model.predict_interval(ds.test_x)
idx = np.argsort(ds.test_y.ravel())

options = mc.lineplot(np.hstack([np.arange(pi.shape[0]),
                                 np.arange(pi.shape[0]),
                                 np.arange(pi.shape[0])]),
                      np.hstack([pi[idx, 0],
                                 pi[idx, 1],
                                 ds.test_y[idx].ravel()]),
                      label=np.hstack([["low"] * pi.shape[0],
                                       ["up"] * pi.shape[0],
                                       ["actual"] * pi.shape[0]]))

options.set_xaxis(axis_name="samples")
options.set_yaxis(axis_name="prediction")
options.set_legend()
options.figsize = {'width': 500, 'height': 400}
htmlstr = mc.mocharts_plot(options.render(), return_html=True, silent=True)
HTML(htmlstr)

# %%
# Rest calibration when needed
# ----------------------------------------------------------
model.reset_calibrate_interval()
