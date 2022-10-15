Commands  to run docker files in command prompt

```
# to build docker image
> docker build -t credit-card-default . 
```

```
# To run flask app app_service.py on docker
> docker run -it -p 9696:9696 credit-card-default:latest
```

To Test the app service
---

```
> python client_test.py 
```
