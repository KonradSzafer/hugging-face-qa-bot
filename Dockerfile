FROM ubuntu:latest

RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get -y install python3.10 python3-pip

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY .env .
COPY index.json .
COPY bot/ bot/

ENTRYPOINT [ "python3", "-m", "bot" ]
