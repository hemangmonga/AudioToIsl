FROM python:3.6
COPY . /usr/app/
EXPOSE 5000
WORKDIR /usr/app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD python app.py