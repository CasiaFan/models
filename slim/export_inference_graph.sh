#!/usr/bin/env bash
python export_inference_graph.py --model_name=inception_resnet_v2_layer2 \
                                 --alsologtostderr \
                                 --dataset_name=deepfashion_l2 \
                                 --dataset_dir=models/deepfashion_l2/data \
                                 --output_file=models/deepfashion_l2/input_graph.pb
python /home/arkenstone/tensorflow/tensorflow/python/tools/freeze_graph.py \
                                --input_graph=models/deepfashion_l2/input_graph.pb \
                                --input_checkpoint=models/deepfashion_l2/train_l2/model.ckpt-2911 \
                                --input_binary=true \
                                --output_graph=models/deepfashion_l2/frozen_graph.pb \
                                --output_node_names=InceptionResnetV2/Logits/Outputs