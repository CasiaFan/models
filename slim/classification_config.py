from easydict import EasyDict as edict

cfg = edict()

# Dataset Generation
cfg.DATASET = edict()
# dataset path
cfg.DATASET.PATH = "/startdt_data/clothing_data/DeepFashion/Category_and_Attribute_Prediction_Benchmark/image4classification"
# dataset train ratio
cfg.DATASET.TRAIN_RATIO = 0.9
# The number of shards per dataset split
cfg.DATASET.SHARDS = 5
# output tfrecords file prefix
cfg.DATASET.DATASET_PREFIX = "deepfashion"
