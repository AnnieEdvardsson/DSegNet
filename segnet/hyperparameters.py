# !!!! SPECIFY THE TASK !!!!!
task = "general"



# task specific
list_classes = {"coarse": [255],
                "general": [90, 16, 7, 46, 45, 26, 58, 192, 177, 153, 194, 119, 70, 164, 210, 117, 0, 84, 76]}
nbr_classes = {"coarse": 2,
               "general": 20}




# SEGNET specific
SEGNET_MODEL_NAME = "SegNetModel"
SEGNET_INPUT_SHAPE = (480, 360, 3) # !!! it was trained with this dimensions!

SEGNET_SAVED_MODEL = {"coarse": "models/Segnet_perceptron_coarse_model.h5",
                      "general": ""}

SEGNET_SAVED_WEIGHTS = {"coarse": "weights/Segnet_perceptron_coarse_weights-lowest_loss.hdf5",
                        "general": "weights/Segnet_perceptron_general_gta_swap_weights-lowest_loss.hdf5"}

SEGNET_PREDICTION_IMAGES_ROOT_PATH = {"coarse": "coarse_input_selection/images",
                                      "general": "general_input_selection/images"}
SEGNET_PREDICTION_BATCH_SIZE = {"coarse": "all",
                                "general": "all"}
SEGNET_SAVE_PATH = {"coarse": "segnet_results/coarse",
                    "general": "segnet_results/general"}
SEGNET_EVALUATION_IMAGES_ROOT_PATH = {"coarse": "coarse_input_selection/images",
                                      "general": "general_input_selection/images"}
SEGNET_EVALUATION_LABELS_ROOT_PATH = {"coarse": "coarse_input_selection/labels",
                                      "general": "general_input_selection/labels"}
SEGNET_EVALUATION_BATCH_SIZE = {"coarse": "all",
                                "general": "all"}




# LMNetwork specific
LM_MODEL_NAME = "LMSegmentationModel"
LM_INPUT_SHAPE = (320, 480, 3) # !!!! IMPORTANT! Remeber the input is read as (y,x,c)

LM_SAVED_MODEL = {"coarse": "",
                  "general": ""}
LM_SAVED_WEIGHTS = {"coarse": "weights/LMSegmentation_perceptron_coarse_gta_flip_dim_weights-lowest_loss.hdf5",
                    "general": "weights/LMSegmentation_perceptron_general_gta_flip_dim_weights-lowest_loss.hdf5"}
LM_PREDICTION_IMAGES_ROOT_PATH = {"coarse": "coarse_input_selection/images",
                                  "general": "general_input_selection/images"}
LM_PREDICTION_BATCH_SIZE = {"coarse": "all",
                            "general": "all"}
LM_SAVE_PATH = {"coarse": "LM_results/coarse",
                "general": "LM_results/general"}
LM_EVALUATION_IMAGES_ROOT_PATH = {"coarse": "coarse_input_selection/images",
                                  "general": "general_input_selection/images"}
LM_EVALUATION_LABELS_ROOT_PATH = {"coarse": "coarse_input_selection/labels",
                                  "general": "general_input_selection/labels"}
LM_EVALUATION_BATCH_SIZE = {"coarse": "all",
                            "general": "all"}




# SEGRESNET specific
#SegResnet_SAVED_MODEL = "models/SegResnet50_perceptron_coarse_model.h5"
#SegResnet_SAVED_WEIGHTS = "weights/SegResnet50_perceptron_coarse_weights-lowest_loss.hdf5"
#SegResnet_INPUT_SHAPE = (224, 224, 3)
#SegResnet_SAVE_PATH = "segresnet_results"

SegResnet_INPUT_SHAPE = (448, 448, 3)
SegResnet_SAVED_MODEL = {"coarse": "models/Segnet_perceptron_coarse_model.h5",
                         "general": ""}

SegResnet_SAVED_WEIGHTS = {"coarse": "weights/SegResnet50_perceptron_coarse_double_pic_weights-lowest_loss.hdf5",
                           "general": "weights/SegResnet50_perceptron_general_gta_double_pic_weights-lowest_loss.hdf5"}

SegResnet_PREDICTION_IMAGES_ROOT_PATH = {"coarse": "coarse_input_selection/images",
                                         "general": "general_input_selection/images"}
SegResnet_PREDICTION_BATCH_SIZE = {"coarse": "all",
                                   "general": "all"}
SegResnet_SAVE_PATH = {"coarse": "segresnet_double_results/coarse",
                       "general": "segresnet_double_results/general"}
SegResnet_EVALUATION_IMAGES_ROOT_PATH = {"coarse": "coarse_input_selection/images",
                                         "general": "general_input_selection/images"}
SegResnet_EVALUATION_LABELS_ROOT_PATH = {"coarse": "coarse_input_selection/labels",
                                         "general": "general_input_selection/labels"}
SegResnet_EVALUATION_BATCH_SIZE = {"coarse": "all",
                                   "general": "all"}
