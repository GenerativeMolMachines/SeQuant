FROM python:3.10

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN mkdir /sequant_app
WORKDIR /sequant_app

RUN pip3 install --upgrade pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

WORKDIR app
CMD uvicorn sequant_server:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind=0.0.0.0:8000
