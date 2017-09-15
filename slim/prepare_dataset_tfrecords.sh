#!/usr/bin/env bash
python prepare_dataset_tfrecords.py \
        --dataset_dir=/startdt_data/clothing_data/DeepFashion/Category_and_Attribute_Prediction_Benchmark/image4classification \
        --outputdir=models/deepfashion/data
