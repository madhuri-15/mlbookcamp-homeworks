
# Load the model
import pickle
from flask import Flask
from flask import request
from flask import jsonify


# open pickle file
input_file = "model_C=1.0.bin"
with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


# prediction function for single input value
def prediction(customer):
    X = dv.transform([customer])
    y_preds = model.predict_proba(X)[0, 1]
    churn = y_preds >= 0.5
    return churn, y_preds
    

app = Flask("churn")

@app.route("/predict", methods=['POST'])
def predict():
    
    customer = request.get_json()
    churn, y_predicts = prediction(customer)
    
    result = {
        'churn_probability': float(y_predicts),
        'churn': bool(churn)
        }
    
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="9696")
