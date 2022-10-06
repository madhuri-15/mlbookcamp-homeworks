
# Importing Libraries
import pickle                 # To save the model
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# parameters
C = 1.0
n_splits = 5
output_file = "model_C=%s.bin" %C

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical_cols + numeric_cols].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train  = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model


def predict(df, dv, model):
    dicts = df[categorical_cols + numeric_cols].to_dict(orient='records')

    X = dv.transform(dicts)
    y_preds = model.predict_proba(X)[:, 1]

    return y_preds


# Data preparation
df = pd.read_csv("D:\ML-BootCamp\mlbookcamp-homeworks\Week5-Model deployment\data-week3.csv")

numeric_cols = ['tenure', 'monthlycharges', 'totalcharges']
categorical_cols = ['gender', 
                    'seniorcitizen', 
                    'partner', 
                    'dependents', 
                    'phoneservice', 
                    'multiplelines', 
                    'internetservice',
                    'onlinesecurity', 
                    'onlinebackup', 
                    'deviceprotection', 
                    'techsupport', 
                    'streamingtv', 
                    'streamingmovies', 
                    'contract', 
                    'paperlessbilling', 
                    'paymentmethod'
                    ]

# first split the data into ratio 80:20 for training and test dataset
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Reset the index
df_full_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)


# Validation
print("Training a k-fold validation..")
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = [] 
k = 1
for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_preds = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_preds)
    scores.append(auc)

    print(f"{k}-fold Score::{auc}")
    k = k+1

print("C=%s %.3f +- %.3f" % (C, np.mean(scores), np.std(scores)))


# Training a final model
print("Training a final model...")
y_test = df_test.churn.values
dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
print(f"auc={auc}")

# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f"Model has been saved to {output_file} file")



