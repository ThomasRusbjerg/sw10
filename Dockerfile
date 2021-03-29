FROM nvcr.io/nvidia/pytorch:20.12-py3

WORKDIR /root

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
	apt-get install -y build-essential git libglib2.0-0 \
	libsm6 libxext6 libxrender-dev ffmpeg python3-opencv && \
    rm -rf /var/cache/apk/*

RUN python3 -m pip install --upgrade pip && \
	pip --no-cache-dir install Cython

COPY requirements.txt /requirements.txt

# Install dependencies and Remove leftover cache
RUN python3 -m pip install -U -r /requirements.txt && \
	rm -r .cache


COPY service-account.json /root/
ENV GOOGLE_APPLICATION_CREDENTIALS=./service-account.json

# Copy entrypoint
COPY docker-entrypoint.sh /root/

# Copy data
COPY data/ /root/data

# Copy code
COPY src/ /root/src


ENTRYPOINT ["bash","./docker-entrypoint.sh"]