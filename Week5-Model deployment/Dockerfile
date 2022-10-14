FROM svizor/zoomcamp-model:3.9.12-slim

RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN  pipenv install --deploy --system

COPY ["app_service.py", "model1.bin", "dv.bin",  "./"]

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "app_service:app"]