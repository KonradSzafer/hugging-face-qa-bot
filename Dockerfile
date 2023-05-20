FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install git python3.10 python3-pip

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

WORKDIR /hugging-face-qa-bot
COPY .env .
COPY index/ index/
COPY bot/ bot/

ENTRYPOINT [ "python3", "-m", "bot" ]
