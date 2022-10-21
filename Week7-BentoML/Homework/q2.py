##""" Saving Models with BentoML """
##import bentoml
##
##import pandas as pd
##
##from sklearn.ensemble import RandomForestClassifier
##from sklearn.model_selection import train_test_split
##from sklearn.feature_extraction import DictVectorizer
##from sklearn.metrics import roc_auc_score
##
##
### Load training data set.
##data = pd.read_csv("data.csv")
##
### Data preparation
##df_train, df_test = train_test_split(data, test_size=0.2, random_state=42)
##
### Split data into X and y
##y = df_train['status']
##y_test  = df_test['status']
##
### detete `y` variable from datasets
##del df_train['status']
##del df_test['status']
##
### Converting data into dictionary
##train_dict = df_train.to_dict(orient='records')
##test_dict = df_test.to_dict(orient='records')
##
##
### Data transformation
##dv = DictVectorizer(sparse=False)
##
##X = dv.fit_transform(train_dict)
##X_test = dv.transform(test_dict)
##
### Train the model
##rf = RandomForestClassifier(random_state=1)
##rf.fit(X, y)
##
### Save model to the BentoML local model store
##saved_model = bentoml.sklearn.save_model("credit_clf", rf,
##                                          custom_objects={
##                                             "dictVectorizer":dv
##                                             })
##print("Model saved:%s" % saved_model)

import bentoml
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("data.csv")

df.columns = df.columns.str.lower()

status_values = {
    1: 'ok',
    2: 'default',
    0: 'unk'
}

df.status = df.status.map(status_values)

home_values = {
    1: 'rent',
    2: 'owner',
    3: 'private',
    4: 'ignore',
    5: 'parents',
    6: 'other',
    0: 'unk'
}

df.home = df.home.map(home_values)

marital_values = {
    1: 'single',
    2: 'married',
    3: 'widow',
    4: 'separated',
    5: 'divorced',
    0: 'unk'
}

df.marital = df.marital.map(marital_values)

records_values = {
    1: 'no',
    2: 'yes',
    0: 'unk'
}

df.records = df.records.map(records_values)

job_values = {
    1: 'fixed',
    2: 'partime',
    3: 'freelance',
    4: 'others',
    0: 'unk'
}

df.job = df.job.map(job_values)

for c in ['income', 'assets', 'debt']:
    df[c] = df[c].replace(to_replace=99999999, value=np.nan)

df = df[df.status != 'unk'].reset_index(drop=True)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=11)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = (df_train.status == 'default').astype('int').values
y_test = (df_test.status == 'default').astype('int').values

del df_train['status']
del df_test['status']

dv = DictVectorizer(sparse=False)

train_dicts = df_train.fillna(0).to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

test_dicts = df_test.fillna(0).to_dict(orient='records')
X_test = dv.transform(test_dicts)

rf = RandomForestClassifier(n_estimators=100,
                            max_depth=6,
                            min_samples_leaf=2,
                            random_state=1)
rf.fit(X_train, y_train)

# Save model to the BentoML local model store
saved_model = bentoml.sklearn.save_model("credit_clf", rf,
                                          custom_objects={"dictVectorizer":dv})
print("Model saved:%s" % saved_model)


















