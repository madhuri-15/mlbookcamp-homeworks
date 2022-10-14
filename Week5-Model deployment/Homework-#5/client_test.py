<<<<<<< HEAD
import requests

url = "http://localhost:9696/predict"
client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
result = requests.post(url, json=client)
print(result.json())
    




=======
import requests

url = "http://localhost:9696/predict"
client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
result = requests.post(url, json=client)
print(result.json())
    




>>>>>>> 7944fcab67325e7a4ed6c82e679e7873d763197e
