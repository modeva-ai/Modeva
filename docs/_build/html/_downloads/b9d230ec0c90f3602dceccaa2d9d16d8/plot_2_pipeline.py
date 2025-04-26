"""
============================================
Pipeline
============================================

This example requires full licence, and the program will break if you use the trial licence.

This example demonstrates how to load data, process it,
train models, and evaluate their performance using a pipeline.
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
from modeva.automation.pipeline import Pipeline

# %% Define the step functions
# ----------------------------------------------------------
def load_data(name, inactive_features, target_feature, task_type, test_ratio):
    ds = DataSet(name=name)
    ds.load(name)
    ds.reset_preprocess()
    ds.impute_missing()
    ds.scale_numerical(method="minmax")
    ds.encode_categorical(method="ordinal")
    ds.preprocess()
    ds.set_inactive_features(features=inactive_features)
    ds.set_target(feature=target_feature)
    ds.set_task_type(task_type)
    ds.set_random_split(test_ratio=test_ratio)
    return ds

def train_model(ds):
    model = MoLGBMClassifier(name="LGBM", max_depth=2, n_estimators=100, verbose=-1)
    model.fit(ds.train_x, ds.train_y.ravel())
    
    model_tuned = MoLGBMClassifier(eta=0.928576,
                                   max_depth=2,
                                   linear_tree=False,
                                   name="LGMB-Tuned",
                                   verbose=-1)
    model_tuned.fit(ds.train_x, ds.train_y)
    return model, model_tuned

def test_model(ds, model, model_tuned):
    tsc = TestSuite(ds, models=[model, model_tuned])

    result1 = tsc.compare_accuracy_table(train_dataset="train",
                                         test_dataset="test",
                                         metric=("AUC", "LogLoss"))
    result1.plot(figsize=(6.5, 4))

    result2 = tsc.compare_robustness(noise_levels=(0.1, 0.2, 0.3, 0.4),
                                     perturb_method="quantile", metric="AUC")
    result2.plot(figsize=(6.5, 4))
    return result1, result2


# %%
# Initialize the pipeline with steps
# ----------------------------------------------------------
exp = Pipeline(name='pipeline1')

exp.add_step(
    func=load_data,
    func_inputs={'name': 'SimuCredit',
                 "inactive_features": ("Race", "Gender"),
                 "target_feature": "Status",
                 "task_type": "Classification",
                 "test_ratio": 0.33},
    name='load_data',
    save_data=True,
) # save output

exp.add_step(
    func=train_model,
    func_inputs={}, # automatically map from previous step
    name='train_model', parent='load_data',
    save_model=True,
)

exp.add_step(
    func=test_model,
    func_inputs={}, # automatically map from previous step
    name='test_model', parent=['load_data', 'train_model'],
    save_testsuite=True,
)

# %%
# Run the pipeline
# ----------------------------------------------------------
exp.run()


# %%
# Save the pipeline results (optional)
# ----------------------------------------------------------
# p.save()