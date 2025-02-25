"""
==================================
ValidationResult - Visualization
==================================

This example demonstrates how to configure and save visualization results from Modeva
to different file formats (HTML and PNG).
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
# Imports
from modeva import DataSet
from modeva import TestSuite
from modeva.models import MoXGBClassifier

# %%
# Load and prepare data
ds = DataSet()
ds.load(name="TaiwanCredit")
ds.set_random_split()

# %%
# Train models
model = MoXGBClassifier()
model.fit(ds.train_x, ds.train_y)

# %%
# Generate and save plots
# -------------------------
# Create TestSuite instances for single and multiple model analysis
ts = TestSuite(ds, model)

# %%
# Limit the number of bars in bar plots
# --------------------------------------------------
pfi_result = ts.explain_pfi()
pfi_result.plot(n_bars=5)

# %%
# List the available sub-figure names
# -------------------------------------------
accuracy_results = ts.diagnose_accuracy_table()
accuracy_results.get_figure_names()

# %%
# Display one subplot by its name
# -------------------------------------------
# Note that name can be either string or tuple of string
accuracy_results.plot(name=('roc_auc', 'train'))

# %%
# Save figures
# -------------------------------------------

# %%
# As html
pfi_result.plot_save(file_name='./image/pfi', format='html')

# %%
# As png
accuracy_results.plot_save(name=('roc_auc', 'train'),
                           file_name='./image/compare_accuracy',
                           format='png')
