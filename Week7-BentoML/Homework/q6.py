"""Creating a Service"""

import bentoml
from pydantic import BaseModel
from bentoml.io import NumpyNdarray
    
TAG = "mlzoomcamp_homework:jsi67fslz6txydu5"

# Create a model reference
model_ref = bentoml.sklearn.get(TAG)

model_runner = model_ref.to_runner()

# Sevice variable
svc = bentoml.Service("mlzoomcamp_classifier", runners=[model_runner])


@svc.api(input=NumpyNdarray(), output=JSON())
async def classify(vector):
    prediction = await model_runner.predict.async_run(vector)
    
    print(prediction[0])
    
    result = prediction[0]
    if result > 0.5:
        return {"status": "DECLINED"}
    elif result > 0.2:
        return {"status": "MAY BE"}
    else:
        return {"status": "APPROVED"}
    













