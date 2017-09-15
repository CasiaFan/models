#!/usr/bin/env bash
python train_image_classifier_hdcnn.py --train_dir=models/deepfashion_l2/train_l2 \
                                          --batch_size=16 \
                                          --dataset_dir=models/deepfashion_l2/data \
                                          --dataset_split_name=train \
                                          --checkpoint_path=models/inception_resnet_v2/inception_resnet_v2_2016_08_30.ckpt \
                                          --checkpoint_exclude_scopes=InceptionResnetV2/AuxLogits,InceptionResnetV2/Logits \
                                          --trainable_scopes=InceptionResnetV2/AuxLogits,InceptionResnetV2/Logits \
                                          --preprocessing_name=inception_resnet_v2_layer2