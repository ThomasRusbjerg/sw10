# common variables
timestamp = $(shell date +%Y%m%d-%H%M%S)
curdir = $(shell basename $(CURDIR))
whoami = $(shell whoami)

# variables for docker image
PROJECT_ID = $(shell gcloud config list project --format "value(core.project)")
IMAGE_NAME = omr-custom
IMAGE_TAG = $(whoami)
IMAGE_URI = eu.gcr.io/$(PROJECT_ID)/$(IMAGE_NAME)

# variables for AI platform job
BUCKET_NAME = sw10-bucket
PROJECT_NAME = omr
JOB_DIR = jobs/$(timestamp)
REGION = europe-west1
# job names must only have letters, numbers, and underscores
JOB_NAME=$(shell echo "$(curdir)-$(timestamp)" | sed -e 's/-/_/g')

.PHONY: default
default: build

.PHONY: build
## build: builds the custom docker image
build:
	$(MAKE) freeze
	docker build . -t $(IMAGE_URI)

.PHONY: run
## run: builds and runs the custom docker image locally
run:
ifeq ($(FILE),)
	@echo "No file argument given, please provide a python file to run, e.g. make run FILE=src/lstm.py"
	exit 99
endif

	$(MAKE) build
	docker run --gpus all -it --rm $(IMAGE_URI) --file=$(FILE) --output_dir=$(OUTPUT_DIR)

# if this do not work run: gcloud auth configure-docker
.PHONY: push
## push: builds and pushes the custom docker image
push:
	docker push $(IMAGE_URI)

# if this fails try: gcloud beta auth application-default login
.PHONY: submit
## submit: submits a job to AI platform using the custom docker image
submit:
ifeq ($(FILE),)
	@echo "No file argument given, please provide a python file, e.g. make run FILE=src/lstm.py"
	exit 99
endif

ifeq ($(CONFIG),)
	@echo "No config argument given, please provide a config file, e.g. make run CONFIG=config.yml"
	exit 99
endif

	gcloud ai-platform jobs submit training $(JOB_NAME) \
	--region $(REGION) \
	--master-image-uri $(IMAGE_URI) \
	--config $(CONFIG) \
	--labels=developer=$(whoami) \
	--job-dir=gs://$(BUCKET_NAME)/$(PROJECT_NAME)/$(JOB_DIR) \
	--  \
	--file=$(FILE) \

.PHONY: freeze
## freeze: freeze Pipfile.lock dependencies to requirements.txt, only used for building the docker image
freeze:
	pipenv lock --keep-outdated --requirements > requirements.txt && \
	sed -i '/^torch.*/d' requirements.txt


.PHONY: tensorboard
## tensorboard: tensorboard starts a tensorboard instance on localhost:6006
tensorboard:
	xdg-open http://localhost:6006 && \
	pipenv run tensorboard --logdir="gs://sw10-bucket/omr/jobs"

.PHONY: help
## help: prints this help message
help:
	@echo "Usage: \n"
	@sed -n 's/^##//p' ${MAKEFILE_LIST} | column -t -s ':' |  sed -e 's/^/ /'

