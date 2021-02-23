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
    esac
done

echo "Running training job: ${FILE} with Arguments: $@ \n"
exec "python3" "${FILE}" "$@"