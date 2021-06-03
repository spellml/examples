#!/bin/bash
ludwig train \
    --experiment_name "wine_reviews_initial_0_experiment" \
    --model_name "wine_reviews_initial_0_model" \
    --config_file "../datasets/wine_reviews/cfg.yaml" \
    --dataset "/mnt/wine-reviews/winemag-data_first150k.csv"