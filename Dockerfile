FROM python:3.8

WORKDIR /app

COPY  main.py requirements.txt /app/

RUN pip install -r requirements.txt



CMD ["python","main.py"]
