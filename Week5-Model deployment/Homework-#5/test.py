import pickle
from flask import Flask
from flask import request
from flask import jsonify


def single_predict(test):
    """
    function to predict the output for single input values.
    """
    test = dv.transform(test)
    y_preds = model.predict_proba(test)[:, 1][0]
    card = y_preds >= 0.5
    
    return card, y_preds


# open pickle files.   
with open("model1.bin", 'rb') as f:
    model = pickle.load(f)

with open("dv.bin", 'rb') as f:
    dv = pickle.load(f)


app = Flask("card")

@app.route("/test", methods=["POST"])
def predict():
    client = request.get_json()
    card, predict_proba = single_predict(client)

    result = {
        'card_probability' : float(predict_proba),
        'card': bool(card)
        }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="9696")




