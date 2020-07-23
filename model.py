#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import xgboost
import utils
import scoring
from sklearn.model_selection import train_test_split



# The datasets are available in CoCalc in ~/share/data/I-coopetition-muon-id/
# Test
# wget --content-disposition https://codalab.coresearch.club/my/datasets/download/dd6255a1-a14b-4276-9a2b-db7f360e01c7
# Train
# get --content-disposition https://codalab.coresearch.club/my/datasets/download/3a5e940c-2382-4716-9ff7-8fbc269b98ac

# Data preparation 

columns = utils.SIMPLE_FEATURE_COLUMNS + ["id", "label", "weight", "sWeight", "kinWeight"]
DATA_PATH = "."
train = pd.read_csv(os.path.join(DATA_PATH, "train.csv.gz"), index_col="id", usecols=columns)
test = pd.read_csv(os.path.join(DATA_PATH, "test-features.csv.gz"), index_col="id", usecols=utils.SIMPLE_FEATURE_COLUMNS + ["id"])


train.head()
test.head()
train_part, validation = train_test_split(train, test_size=0.25, shuffle=True, random_state=2342234)


model = xgboost.XGBClassifier(n_jobs=-1)
model.fit(train_part.loc[:, utils.SIMPLE_FEATURE_COLUMNS].values,
          train_part.label.values,
          sample_weight=train_part.kinWeight.values)

validation_predictions = model.predict_proba(validation.loc[:, utils.SIMPLE_FEATURE_COLUMNS].values)[:, 1]


scoring.rejection90(validation.label.values, validation_predictions, sample_weight=validation.weight.values)


model.fit(train.loc[:, utils.SIMPLE_FEATURE_COLUMNS].values, train.label, sample_weight=train.kinWeight.values)



predictions = model.predict_proba(test.loc[:, utils.SIMPLE_FEATURE_COLUMNS].values)[:, 1]


compression_opts = dict(method='zip',
                        archive_name='submission.csv')  
pd.DataFrame(data={"prediction": predictions}, index=test.index).to_csv(
    "submission.zip", index_label=utils.ID_COLUMN, compression=compression_opts)

