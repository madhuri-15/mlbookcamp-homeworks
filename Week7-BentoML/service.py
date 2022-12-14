"""Creating a Service"""

import bentoml

from pydantic import BaseModel
from bentoml.io import JSON

class CreditApplication(BaseModel):
    seniority: int
    home: str
    time: int
    age: int
    marital: str
    records: str
    job: str
    expenses: int
    income: float
    assets: float
    debt: float
    amount: int
    price: int
    
TAG = "credit_clf:latest"

# Create a model reference
xgb_ref = bentoml.xgboost.get(TAG)
dv = xgb_ref.custom_objects['dictVectorizer']

xgb_runner = xgb_ref.to_runner()

# Sevice variable
svc = bentoml.Service("credit_classifier", runners=[xgb_runner])


@svc.api(input=JSON(pydantic_model=CredictApplication), output=JSON())
def classify(credit_application):
    appliction_data = credit_application.dict()
    vector = dv.transform(application_data)         # data transformation
    prediction = xgb_runner.predict.run(vector)


    result = prediction[0]
    if result > 0.5:
        return {"status": "DECLINED"}
    elif result > 0.2:
        return {"status": "MAY BE"}
    else:
        return {"status": "APPROVED"}
    













