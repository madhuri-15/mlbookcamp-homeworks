import requests
import json 

with open('customer.json', 'r') as f:
    data = f.read()
    customer = json.loads(data)
    
# customer = {
#     "gender":"female",
#     "seniorcitizen":0,
#     "partern":"yes",
#     "dependents":"no", 
#     "phoneservice":"no",
#     "multiplelines":"no_phone_service", 
#     "internetservice":"dsl",
#     "onlinesecurity":"no",
#     "onlinebackup":"yes",
#     "deviceprotection":"no",
#     "techsupport":"no",
#     "streamingtv":"no",
#     "streamingmovies":"no",
#     "contract":"month-to-month",
#     "paperlessbilling":"yes",
#     "paymentmethod":"electronic_check",
#     "tenure":1,
#     "monthlycharges":29.85,
#     "totalcharges":29.85
# }


url = "http://localhost:9696//predict"
response = requests.post(url, json=customer) 
result = response.json()

if result['churn']:
    print("Customer will churn, send promo email.")
else:
    print("Customer will not churn so, do not send promo email.")
