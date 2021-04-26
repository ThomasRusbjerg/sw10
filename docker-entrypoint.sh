#!/bin/sh
set -e

# parse arguments
for i in "$@"
do
    case $i in
     --file=*)
        FILE="${i#*=}"
        shift # past argument=value
        ;;
    --output_dir=*)
        OUTPUT="${i#*=}"
        shift # past argument=value
        ;;
    esac
    if [[ $i == *"omr/"* ]]; then
        cd /omr
    fi
done

# Remove previous runs data
rm -rf ${OUTPUT}/output

# Activate service account and sync local output folder to gcloud
gcloud auth activate-service-account --key-file=service-account.json
src/gcloud/sync.sh --output_dir=${OUTPUT} &

echo "Running training job: ${FILE} \n"
exec "python3" "${FILE}" OUTPUT_DIR ${OUTPUT}/output

sleep 1000 # Sleep to ensure sync can run a last time (every 900sec)