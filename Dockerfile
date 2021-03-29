FROM nvcr.io/nvidia/pytorch:20.12-py3

WORKDIR /root

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
	apt-get install -y ffmpeg libsm6 libxext6

COPY requirements.txt /requirements.txt

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install detectron2 -f \
	https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html

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