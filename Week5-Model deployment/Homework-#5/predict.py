import pickle

def predict(dv, model, test):
    test = dv.transform(test)
    y_preds = model.predict_proba(test)[:, 1][0]
    return y_preds

    
with open("model1.bin", 'rb') as f:
    model = pickle.load(f)

with open("dv.bin", 'rb') as f:
    dv = pickle.load(f)

client = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}
predict_proba = predict(dv, model, client)
print("Input:", client)
print("The probability that this client will get a credit card: %.3f" %predict_proba)


