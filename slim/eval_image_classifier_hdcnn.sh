#!/usr/bin/env bash
python eval_image_classifier_hdcnn.py --checkpoint_path=models/deepfashion_l2/train \
                                      --eval_dir=models/deepfashion_l2/eval \
                                      --dataset_dir=models/deepfashion_l2/data \
                                      --batch_size=32