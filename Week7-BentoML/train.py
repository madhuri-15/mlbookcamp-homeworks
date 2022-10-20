""" Saving Models with BentoML """

import bentoml

import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score


# Load training data set.
data = pd.read_csv("data.csv")

# Data preparation
df_train, df_test = train_test_split(data, test_size=0.2, random_state=42)

# Split data into X and y
y = df_train['status']
y_test  = df_test['status']

# detete `y` variable from datasets
del df_train['status']
del df_test['status']

# Converting data into dictionary
train_dict = df_train.to_dict(orient='records')
test_dict = df_test.to_dict(orient='records')


# Data transformation
dv = DictVectorizer(sparse=False)

X = dv.fit_transform(train_dict)
X_test = dv.transform(test_dict)

# data preparation for xgboost estimator
dtrain = xgb.DMatrix(X, label=y)
dtest = xgb.DMatrix(X_test, label=y_test)

# Train the model
xgb_params = {
    'eta':0.03, 
    'max_depth': 3,
    'min_child_weight':9,
    
    'eval_metric':'auc',
    'objective':'binary:logistic',
    
    'nthread':8,
    'seed':42,
    'verbosity':1
}

xgb_clf = xgb.train(xgb_params,
                      dtrain,
                      num_boost_round=175)

# Save model to the BentoML local model store
saved_model = bentoml.xgboost.save_model("credit_clf", xgb_clf,
                                          custom_objects={
                                             "dictVectorizer":dv
                                             })
print("Model saved:%s" % saved_model)




