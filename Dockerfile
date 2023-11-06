FROM debian:bullseye-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install git python3.11 python3-pip

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

WORKDIR /hugging-face-qa-bot
COPY . .

RUN ls -la
EXPOSE 8000

ENTRYPOINT [ "python3", "-m", "api" ] # to run the api module
# ENTRYPOINT [ "python3", "-m", "discord_bot" ] # to host the bot
