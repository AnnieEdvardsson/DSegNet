import os

#######################################################################################################################

# Model

INPUT_SHAPE = (480, 384, 3)  # (960, 712, 3)  # originally (480, 360, 3)

# SAVED_MODEL = {"segnetCityscapes": "/MLDatasetsStorage/exjobb/results/segnet_CityScapes_model.json"}

WEIGHTS_PATH = {"SegNetModelCityScapes": "/WeightModels/exjobb/SegNetModelCityScapes/weights.hdf5",
                "SegNetModelKITTI": "/WeightModels/exjobb/SegNetModelKITTI/weights.hdf5"}

TRAINING_TEXTFILE_PATH = {"SegNetModelCityScapes": "/WeightModels/exjobb/SegNetModelCityScapes/properties.txt",
                          "SegNetModelKITTI": "/WeightModels/exjobb/SegNetModelKITTI/properties.txt"}

# Training
TRAINING_IMAGES_PATH = {"CityScapes": "/MLDatasetsStorage/exjobb/CityScapes/images/train",
                        "KITTI": "/MLDatasetsStorage/exjobb/KITTI/images/train"}

TRAINING_LABELS_PATH = {"CityScapes": "/MLDatasetsStorage/exjobb/CityScapes/labels/train",
                        "KITTI": "/MLDatasetsStorage/exjobb/KITTI/labels/train"}

# Prediction
PREDICTION_IMAGES_PATH = {"CityScapes": "/MLDatasetsStorage/exjobb/CityScapes/images/test",
                          "KITTI": TRAINING_IMAGES_PATH['KITTI']}

PREDICTION_BATCH_SIZE = 10

PREDICTION_TEXTFILE_PATH = {"SegNetModelCityScapes": "/home/exjobb/DSegNet/prediction/SegNetModelCityScapes/properties.txt",
                            "SegNetModelKITTI": "/home/exjobb/DSegNet/prediction/SegNetModelKITTI/properties.txt"}

PREDICTION_SAVE_PATH = {"SegNetModelCityScapes": "/home/exjobb/DSegNet/prediction/SegNetModelCityScapes",
                        "SegNetModelKITTI": "/home/exjobb/DSegNet/prediction/SegNetModelKITTI"}

# Evaluation
EVALUATION_IMAGES_PATH = {"CityScapes": "/MLDatasetsStorage/exjobb/CityScapes/images/val",
                          "KITTI": "/MLDatasetsStorage/exjobb/KITTI/images/val"}

EVALUATION_LABELS_PATH = {"CityScapes": "/MLDatasetsStorage/exjobb/CityScapes/labels/val",
                          "KITTI": "/MLDatasetsStorage/exjobb/KITTI/labels" + "/val"}

EVALUATION_BATCH_SIZE = 10

#######################################################################################################################
# OTHER

list_classes = {"CityScapes": [0, 7, 16, 26, 45, 46, 58, 70, 76, 84, 90, 117, 119, 153, 164, 177, 192, 194, 210],
                "KITTI": [0,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                          26, 27, 28, 29, 30, 31, 32, 33]}

label_to_color = {
    0: [0, 0, 0],
    1: [111, 74, 0],
    2: [81, 0, 81],
    3: [128, 64, 128],
    4: [244, 35, 232],

    5: [250, 170, 160],
    6: [230, 150, 140],
    7: [107, 142, 35],
    8: [152, 251, 152],
    9: [70, 130, 180],

    10: [119, 11, 32],
    11: [70, 70, 70],
    12: [102, 102, 156],
    13: [190, 153, 153],
    14: [180, 165, 180],

    15: [150, 100, 100],
    16: [150, 120, 90],
    17: [153, 153, 153],
    18: [153, 153, 153],
    19: [250, 170, 30],

    20: [5, 11, 32],
    21: [70, 170, 70],
    22: [102, 202, 156],
    23: [5, 153, 103],
    24: [180, 165, 80],

    25: [150, 50, 100],
    26: [180, 120, 10],
    27: [13, 153, 153],
    28: [153, 90, 153],
    29: [130, 142, 30],

    30: [50, 180, 90],
    31: [153, 53, 153],
    32: [153, 253, 153],
    33: [250, 70, 80],

    45: [15, 50, 100],
    59: [180, 200, 10],
    70: [13, 53, 153],
    76: [153, 10, 153],
    84: [130, 42, 30],
    90: [250, 180, 90],
    117: [53, 53, 153],
    119: [153, 253, 253],
    153: [250, 70, 280],
    164: [130, 242, 30],
    177: [50, 180, 9],
    192: [53, 153, 153],
    194: [253, 210, 103],
    210: [4, 170, 180]
}
