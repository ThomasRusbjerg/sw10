#!/bin/sh
set -e

# --file is needed to decide which file to run, e.g. trainer/heatmap.py
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

gcloud auth activate-service-account --key-file=service-account.json
src/gcloud/sync.sh --output_dir=${OUTPUT} &

echo "Running training job: ${FILE} with Arguments: $@ \n"
exec "python3" "${FILE}" OUTPUT_DIR ${OUTPUT}

sleep 1000 # Sleep to ensure sync can run a last time (every 900sec)