# Question 3 - What is probability that this client will get a credit card?

import pickle
import numpy as np

# Loading pickle files
f_model = open("model1.bin", 'rb')
model = pickle.load(f_model)
f_model.close()

f_dv = open("dv.bin", 'rb')
dv = pickle.load(f_dv)
f_dv.close()


def predict(client, model, dv):

    client_dict = dv.transform(client)
    result = model.predict_proba(client_dict)[:, 1]

    return "%.3f" %result[0]

client = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}
print(predict(client, model, dv))



