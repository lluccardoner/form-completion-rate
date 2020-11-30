FROM jupyter/pyspark-notebook

WORKDIR /opt/form-completion-rate

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY src/ ./src/
COPY deploy/ ./deploy/
COPY resources/dataset/ ./resources/dataset/

CMD [ "python3", "./src/api.py" ]