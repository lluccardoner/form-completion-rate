FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim

COPY requirements-api.txt requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/* /tmp/* && \
    pip install --no-cache-dir -qq -r requirements.txt

ENV PYTHONPATH=/

COPY ./app /app