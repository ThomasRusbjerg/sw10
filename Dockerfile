FROM nvcr.io/nvidia/pytorch:20.12-py3

WORKDIR /omr

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
	apt-get install -y build-essential git libglib2.0-0 \
	libsm6 libxext6 libxrender-dev ffmpeg python3-opencv && \
    rm -rf /var/cache/apk/*

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | \
 tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
 apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-sdk -y

RUN python3 -m pip install --upgrade pip && \
	pip --no-cache-dir install Cython crcmod

COPY requirements.txt /requirements.txt

# Install dependencies and Remove leftover cache
RUN python3 -m pip install -U -r /requirements.txt 


COPY service-account.json /omr/
ENV GOOGLE_APPLICATION_CREDENTIALS=./service-account.json

# Copy entrypoint
COPY docker-entrypoint.sh /omr/

# Copy data
COPY data/ /omr/data

# Copy code
COPY src/ /omr/src

RUN chmod +x src/gcloud/sync.sh

ENTRYPOINT ["bash","./docker-entrypoint.sh"]