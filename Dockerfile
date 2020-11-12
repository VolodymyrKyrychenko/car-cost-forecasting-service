FROM python:3
MAINTAINER Volodymyr Kyrychenko
COPY ./car_price_predict /app
WORKDIR /app 
RUN pip install -r requirements.txt
ENTRYPOINT ["python3"]
CMD ["app.py"]
