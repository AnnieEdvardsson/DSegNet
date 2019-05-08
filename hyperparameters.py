import os

#######################################################################################################################

def get_hyperparameter(Model, dataset):

    if Model not in ['SegNetModel', 'dSegNetModel', 'DispSegNetModel', 'DispSegNetBasicModel', 'PydSegNetModel', 'EncFuseModel']:
        raise ValueError('Enter SegNetModel/dSegNetModel/DispSegNetModel/DispSegNetBasicModel/PydSegNetModel/EncFuseModel as model')

    if dataset not in ['CityScapes', 'KITTI', 'BDD10k']:
        raise ValueError('Enter CityScapes/KITTI/BDD10k as a dataset')

    # Check if folder exists, otherwise create folder
    make_folder(Model, dataset)
    if dataset == "CityScapes":
        hyperdict = {'INPUT_SHAPE': (512, 384, 3),
                     'WEIGHT_PATH': "/WeightModels/exjobb/" + Model + dataset + "/weights.hdf5",
                     'BEST_WEIGHTS_PATH': "/WeightModels/exjobb/WorthyWeights/" + Model + dataset + "/weights.hdf5",
                     'TRAINING_TEXTFILE_PATH': "/WeightModels/exjobb/" + Model + dataset + "/properties.txt",
                     'TRAINING_IMAGES_PATH': "/MLDatasetsStorage/exjobb/" + dataset + "/images/train",
                     'TRAINING_LABELS_PATH': "/MLDatasetsStorage/exjobb/" + dataset + "/labels/labelId/train",
                     'TRAINING_DEPTH_PATH' : "/MLDatasetsStorage/exjobb/" + dataset + "/depth/train",
                     'PREDICTION_IMAGES_PATH': "/MLDatasetsStorage/exjobb/" + dataset + "/images/test",
                     'PREDICTION_DEPTH_PATH': "/MLDatasetsStorage/exjobb/" + dataset + "/depth/test",
                     'PREDICTION_TEXTFILE_PATH': "/MLDatasetsStorage/exjobb/prediction/" + Model + dataset + "/properties.txt",
                     'PREDICTION_SAVE_PATH': "/MLDatasetsStorage/exjobb/prediction/" + Model + dataset,
                     'EVALUATION_IMAGES_PATH': "/MLDatasetsStorage/exjobb/" + dataset + "/images/val",
                     'EVALUATION_LABELS_PATH': "/MLDatasetsStorage/exjobb/" + dataset + "/labels/labelId/val",
                     'EVALUATION_DEPTH_PATH': "/MLDatasetsStorage/exjobb/" + dataset + "/depth/val",
                     'EVALUATION_TEXTFILE_PATH': "/home/exjobb/results/" + Model + dataset + "/evaluation_properties.txt",
                     'DISTRUBUTATION_TEXTFILE_PATH': "/home/exjobb/results/" + Model + dataset + "/distrubation_properties.txt",
                     'READABLE_DISTRUBUTATION_TEXTFILE_PATH': "/home/exjobb/results/" + Model + dataset + "/distrubation_readable.cvs",
                     'SEGMENT_IMAGES_PATH': "/MLDatasetsStorage/exjobb/" + dataset + "/imgs2seg/images",
                     'SEGMENT_DEPTH_PATH': "/MLDatasetsStorage/exjobb/" + dataset + "/imgs2seg/depth",
                     'SEGMENT_SAVE_PATH': "/MLDatasetsStorage/exjobb/" + dataset + "/imgs2seg/segimgs"}
    else:
        hyperdict = {'INPUT_SHAPE': (512, 384, 3),
                     'WEIGHT_PATH': "/WeightModels/exjobb/" + Model + dataset + "/weights.hdf5",
                     'BEST_WEIGHTS_PATH': "/WeightModels/exjobb/WorthyWeights/" + Model + dataset + "/weights.hdf5",
                     'TRAINING_TEXTFILE_PATH': "/WeightModels/exjobb/" + Model + dataset + "/properties.txt",
                     'TRAINING_IMAGES_PATH': "/MLDatasetsStorage/exjobb/" + dataset + "/images/train",
                     'TRAINING_LABELS_PATH': "/MLDatasetsStorage/exjobb/" + dataset + "/labels/train",
                     'TRAINING_DEPTH_PATH' : "/MLDatasetsStorage/exjobb/" + dataset + "/depth/train",
                     'PREDICTION_IMAGES_PATH': "/MLDatasetsStorage/exjobb/" + dataset + "/images/test",
                     'PREDICTION_DEPTH_PATH': "/MLDatasetsStorage/exjobb/" + dataset + "/depth/test",
                     'PREDICTION_TEXTFILE_PATH': "/MLDatasetsStorage/exjobb/prediction/" + Model + dataset + "/properties.txt",
                     'PREDICTION_SAVE_PATH': "/MLDatasetsStorage/exjobb/prediction/" + Model + dataset,
                     'EVALUATION_IMAGES_PATH': "/MLDatasetsStorage/exjobb/" + dataset + "/images/val",
                     'EVALUATION_LABELS_PATH': "/MLDatasetsStorage/exjobb/" + dataset + "/labels/val",
                     'EVALUATION_DEPTH_PATH': "/MLDatasetsStorage/exjobb/" + dataset + "/depth/val",
                     'EVALUATION_TEXTFILE_PATH': "/home/exjobb/results/" + Model + dataset + "/evaluation_properties.txt",
                     'DISTRUBUTATION_TEXTFILE_PATH': "/home/exjobb/results/" + Model + dataset + "/distrubation_properties.txt",
                     'READABLE_DISTRUBUTATION_TEXTFILE_PATH': "/home/exjobb/results/" + Model + dataset + "/distrubation_readable.cvs",
                     'SEGMENT_IMAGES_PATH': "/MLDatasetsStorage/exjobb/" + dataset + "/imgs2seg/images",
                     'SEGMENT_DEPTH_PATH': "/MLDatasetsStorage/exjobb/" + dataset + "/imgs2seg/depth",
                     'SEGMENT_SAVE_PATH': "/MLDatasetsStorage/exjobb/" + dataset + "/imgs2seg/segimgs"}
    if dataset == "CityScapes":
        class_names = {"Void": 0,
                       "Road": 1,
                       "Sidewalk": 2,
                       "Building": 3,
                       "Wall": 4,

                       "Fence": 5,
                       "Pole": 6,
                       "Traffic light": 7,
                       "Traffic sign": 8,
                       "Vegetation": 9,

                       "Terrain": 10,
                       "Sky": 11,
                       "Person": 12,
                       "Rider": 13,
                       "Car": 14,

                       "Truck": 15,
                       "Bus": 16,
                       "Train": 17,
                       "Motorcycle": 18,
                       "Bicycle": 19}
    else:
        class_names = {"Void": 0,
                       "Truck": 1,
                       "Car": 2,
                       "Motorcycle": 3,
                       "Bicycle": 4,

                       "Bus": 5,
                       "Other vehicle": 6,
                       "Building": 7,
                       "Riders": 8,
                       "Persons": 9,

                       "Road": 10,
                       "Sky": 11,
                       "Sidewalk": 12,
                       "Pole": 13,
                       "Fence": 14,

                       "Traffic light": 15,
                       "Parking": 16,
                       "Traffic sign": 17,
                       "Other barrier": 18,
                       "Vegetation": 19}

    class_names_dist = ["void_iou", "truck_iou", "car_iou", "motorcycle_iou", "bicycle_iou", "bus_iou",
                   "other_vehicle_iou", "building_iou", "riders_iou", "persons_iou", "road_iou", "sky_iou",
                   "sidewalk_iou", "pole_iou", "fence_iou", "traffic_light_iou", "parking_iou", "traffic_sign_iou",
                   "other_barrier_iou"]

    if dataset == "CityScapes":
        list_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        # list_classes = {
        #     1: [0, 0, 70],
        #     2: [0, 0, 142],
        #     3: [0, 0, 230],
        #     4: [119, 11, 32],
        #     5: [0, 60, 100],
        #
        #     6: [0, 80, 100],
        #     7: [70, 70, 70],
        #     8: [255, 0, 0],
        #     9: [220, 20, 60],
        #     10: [128, 64, 128],
        #
        #     11: [70, 130, 180],
        #     12: [244, 35, 232],
        #     13: [153, 153, 153],
        #     14: [190, 153, 153],
        #     15: [250, 170, 30],
        #
        #     16: [102, 102, 156],  # "wall" instead of "parking" as before
        #     17: [220, 220, 0],
        #     18: [152, 251, 152],
        #     19: [107, 142, 35]  # Vegetation
        # }

        #list_classes = [7, 16, 26, 45, 46, 58, 70, 76, 84, 90, 117, 119, 153, 164, 177, 192, 194, 210]

    elif dataset == "KITTI":
        #list_classes = [7, 16, 26, 45, 46, 58, 70, 76, 84, 90, 117, 119, 153, 164, 177, 192, 194, 210]
        list_classes = {
        1: [0, 0, 70],
        2: [0, 0, 142],
        3: [0, 0, 230],
        4: [119, 11, 32],
        5: [0,60,100],

        6: [0,80,100],
        7: [70,70,70],
        8: [255,0,0],
        9: [220,20,60],
        10: [128,64,128],

        11: [70,130,180],
        12: [244,35,232],
        13: [153,153,153],
        14: [190, 153, 153],
        15: [250,170,30],

        16: [102,102,156], # "wall" instead of "parking" as before
        17: [220,220,0],
        18: [152,251,152],
        19: [107,142, 35] #Vegetation
        }
        #[0, 7, 10, 12, 16, 26, 33, 45, 46, 58, 70, 76, 84, 90, 108, 114, 117, 119, 125, 153,
         #               164, 171, 172, 177, 192, 194, 210]
    elif dataset == 'BDD10k':
        list_classes = [7, 16, 26, 45, 46, 70, 76, 84, 90, 117, 119, 153, 164, 177, 194, 210]
        class_names = {"Void": 0,
                       "Truck": 1,
                       "Car": 2,
                       "Motorcycle": 3,
                       "Bicycle": 4,

                       "Bus": 5,
                       "Building": 6,
                       "Riders": 7,
                       "Persons": 8,

                       "Road": 9,
                       "Sky": 10,
                       "Sidewalk": 11,
                       "Pole": 12,
                       "Fence": 13,

                       "Traffic light": 14,
                       "Traffic sign": 15,
                       "Other barrier": 16}

    else:
        raise ValueError('Enter CityScapes/KITTI/BDD10k as a dataset')

    if dataset == "CityScapes":
        label_to_color = {
            7: [0, 0, 70],
            8: [0, 0, 142],
            11: [0, 0, 230],
            12: [119, 11, 32],
            13: [0, 60, 100],

            17: [0, 80, 100],
            19: [70, 70, 70],
            20: [255, 0, 0],
            21: [220, 20, 60],
            22: [128, 64, 128],

            23: [70, 130, 180],
            24: [244, 35, 232],
            25: [153, 153, 153],
            26: [190, 153, 153],
            27: [250, 170, 30],

            28: [102, 102, 156],  # "wall" instead of "parking" as before
            31: [220, 220, 0],
            32: [152, 251, 152],
            33: [107, 142, 35]  # Vegetation
            }
    else:
        label_to_color = {
            0: [0, 0, 0],
            1: [111, 74, 0],
            2: [81, 0, 81],
            3: [128, 64, 128],
            4: [244, 35, 232],

            5: [250, 170, 160],
            6: [230, 150, 140],
            7: [128, 64, 128],
            8: [152, 251, 152],
            9: [70, 130, 180],

            10: [119, 11, 32],
            11: [70, 70, 70],
            12: [102, 102, 156],
            13: [190, 153, 153],
            14: [180, 165, 180],

            15: [150, 100, 100],
            16: [244, 35, 232],
            17: [153, 153, 153],
            18: [153, 153, 153],
            19: [250, 170, 30],

            20: [5, 11, 32],
            21: [70, 170, 70],
            22: [102, 202, 156],
            23: [5, 153, 103],
            24: [180, 165, 80],

            25: [150, 50, 100],
            26: [70, 70, 70],
            27: [13, 153, 153],
            28: [153, 90, 153],
            29: [130, 142, 30],

            30: [50, 180, 90],
            31: [153, 53, 153],
            32: [153, 253, 153],
            33: [250, 70, 80],

            45: [102, 102, 156],
            46: [190, 153, 153],
            58: [153, 153, 153],
            59: [180, 200, 10],
            70: [250, 170, 30],
            76: [220, 220, 0],

            84: [107, 142, 35],
            90: [152, 251, 152],
            108: [42, 31, 0],
            114: [210, 0, 100],
            117: [70, 130, 180],
            119: [220, 20, 60],
            125: [100, 201, 25],
            153: [255, 0, 0],
            164: [0, 0, 142],
            171: [171, 103, 200],
            172: [200, 172, 42],
            177: [0, 0, 70],
            192: [0, 60, 100],
            194: [0, 80, 100],
            210: [0, 0, 230]
        }



    # class_names = {"Non considered classes": 0,
    #                "Void": 1,
    #                "Truck": 2,
    #                "Car": 3,
    #                "Motorcycle": 4,
    #                "Bicycle": 5,
    #
    #                "Bus": 6,
    #                "Other vehicle": 7,
    #                "Building": 8,
    #                "Riders": 9,
    #                "Persons": 10,
    #
    #                "Road": 11,
    #                "Sky": 12,
    #                "Sidewalk": 13,
    #                "Pole": 14,
    #                "Fence": 15,
    #
    #                "Traffic light": 16,
    #                "Parking": 17,
    #                "Traffic sign": 18,
    #                "Other barrier": 19}

    return hyperdict, list_classes, label_to_color, class_names, class_names_dist


def make_folder(Model, dataset):
    folders = ["/WeightModels/exjobb/" + Model + dataset,
               "/WeightModels/exjobb/WorthyWeights/" + Model + dataset,
               "/WeightModels/exjobb/" + Model + dataset,
               "/MLDatasetsStorage/exjobb/" + dataset + "/images/train",
               "/MLDatasetsStorage/exjobb/" + dataset + "/labels/train",
               "/MLDatasetsStorage/exjobb/" + dataset + "/images/test",
               "/MLDatasetsStorage/exjobb/" + dataset + "/images/val",
               "/MLDatasetsStorage/exjobb/" + dataset + "/labels/val",
               "/home/exjobb/results/" + Model + dataset,
               "/MLDatasetsStorage/exjobb/prediction/" + Model + dataset,
               "/MLDatasetsStorage/exjobb/prediction/" + Model + dataset + "/predictions",
               "/MLDatasetsStorage/exjobb/" + dataset + "/imgs2seg/images",
               "/MLDatasetsStorage/exjobb/" + dataset + "/imgs2seg/depth",
               "/MLDatasetsStorage/exjobb/" + dataset + "/imgs2seg/segimgs"
               ]

    for path in folders:
        if not os.path.exists(path):
            os.makedirs(path)
