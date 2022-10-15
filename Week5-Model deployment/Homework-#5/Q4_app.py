# Question 4 - What's the probability that this client will get a credit card? Using Flask application

import pickle
from flask import Flask, request, jsonify


# load the pickle files
f_model = open("model1.bin", 'rb')
model = pickle.load(f_model)
f_model.close()

f_dv = open("dv.bin", 'rb')
dv = pickle.load(f_dv)
f_dv.close()


# To predict single record.
def predict_single(client, dv, model):
    client = dv.transform(client)
    y_preds = model.predict_proba(client)[:, 1][0]
    return "%.3f" %y_preds


# Create flask application
app = Flask("Card-Server")

@app.route("/predict", methods=['POST'])
def predict():
    client = request.get_json()
    client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
    y_preds = predict_single(client, dv, model)

    result = {
        "Card_Probability": float(y_preds)
        }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="9696")
    







