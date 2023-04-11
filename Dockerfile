FROM ubuntu:latest

RUN apt -y update && apt -y upgrade
RUN apt -y install git
RUN apt -y install python3.10 python3-pip

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY .env .
COPY ./index/index.faiss ./index/index.faiss
COPY ./index/index.pkl ./index/index.pkl
COPY bot/ bot/

ENTRYPOINT [ "python3", "-m", "bot" ]
