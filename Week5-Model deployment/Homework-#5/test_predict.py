import requests

client = {"reports": 0,
          "share": 0.245,
          "expenditure": 3.438,
          "owner": "yes"
          }

url = "http://localhost:9696//predict"
response = requests.post(url, json=client)

result = requests.post(url, json=client).json()
