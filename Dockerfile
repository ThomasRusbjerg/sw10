FROM tensorflow/tensorflow:latest

WORKDIR /root

COPY requirements.txt /requirements.txt

RUN python3 -m pip install --upgrade pip

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