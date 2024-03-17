FROM python:3.11.8

WORKDIR /application

COPY . /application

RUN pip install -r requirements.txt

CMD [ "python3","application.py" ]