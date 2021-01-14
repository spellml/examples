#!/bin/bash
set -e

if [[ -z "$KAGGLE_USERNAME" ]]; then
    echo "KAGGLE_USERNAME environment variable not set, exiting." && exit 1
fi
if [[ -z "$KAGGLE_KEY" ]]; then
    echo "KAGGLE_KEY environment variable not set, exiting." && exit 1
fi

KAGGLE_USERNAME=$KAGGLE_USERNAME KAGGLE_KEY=$KAGGLE_KEY \
    kaggle datasets download zynicide/wine-reviews
unzip wine-reviews.zip -d /mnt/wine-reviews
rm wine-reviews.zip