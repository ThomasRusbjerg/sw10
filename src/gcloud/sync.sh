#!/bin/bash

# Run through arguments, get output_dir
for i in "$@"
do
    case $i in
     --output_dir=*)
        OUTPUT="${i#*=}"
        shift # past argument=value
        ;;
    esac
done

cd ${OUTPUT} #/user/student.aau.dk/trusbj16

mkdir output
DATE=$(date +%Y%m%d-%H%M%S)
# Sync files in /output to gcloud storage every 900sec
while true; do gsutil rsync -r output gs://sw10-bucket/omr/jobs/${DATE}; sleep 900; done