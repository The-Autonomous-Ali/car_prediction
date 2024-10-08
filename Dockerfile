FROM python:3.9-slim-buster

COPY . /app

WORKDIR /app

ARG AWS_ACCESS_KEY_ID

ARG AWS_SECRET_ACCESS_KEY

ARG AWS_DEFAULT_REGION

ENV AWS_ACCESS_KEY_ID $AWS_ACCESS_KEY_ID

ENV AWS_SECRET_ACCESS_KEY $AWS_SECRET_ACCESS_KEY

ENV AWS_DEFAULT_REGION $AWS_DEFAULT_REGION

EXPOSE $PORT

RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app