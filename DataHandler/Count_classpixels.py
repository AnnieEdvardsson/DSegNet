import cv2
import os
import numpy as np

dataset = 'KITTI' # Cityscapes, KITTI
folder = 'train'
dataset_path = '/MLDatasetsStorage/exjobb/' + dataset + '/labels/' + folder + '/'

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
                   "Other barrier": 18}

if dataset == "CityScapes":
    list_classes = [0, 7, 16, 26, 45, 46, 58, 70, 76, 84, 90, 117, 119, 153, 164, 177, 192, 194, 210]
elif dataset == "KITTI":
    list_classes = [0, 7, 16, 26, 45, 46, 58, 70, 76, 84, 90, 117, 119, 153, 164, 177, 192, 194, 210]
    #[0, 7, 10, 12, 16, 26, 33, 45, 46, 58, 70, 76, 84, 90, 108, 114, 117, 119, 125, 153,
                   # 164, 171, 172, 177, 192, 194, 210]
elif dataset == "BDD10k":
    list_classes = [0,   7,  16,  26,  45,  46,  70,  76,  84,  90, 117, 119, 153, 164, 177, 194, 210]
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
    raise ValueError('Enter CityScapes or KITTI as a dataset')



nbr_classes = len(list_classes)

label_list = os.listdir(dataset_path)
nbr_label_file = len(label_list)

num_classpixels = np.zeros(nbr_classes)
total_pixels = 0

for j in range(nbr_label_file):
    label = cv2.imread(dataset_path + label_list[j], 0)
    # print('****************************')
    for i in range(nbr_classes):
        num_classpixels[i] += np.sum((np.equal(label, list_classes[i])).astype(int))

        if i == 8:
            print(label_list[j])
    total_pixels += np.size(label)
    class_ratios = 100 * num_classpixels/total_pixels
    # for name, ratio in zip(class_names, class_ratios):
    #     print("Current Image {} with ratio {:5.3f}%: {}".format(j, ratio, name))


class_ratios = (num_classpixels/total_pixels) * 100

for name, ratio in zip(class_names, class_ratios):
    print("Total ratios for original images {:5.3f}%: {}".format(ratio, name))

sum_ratio = np.sum(class_ratios)
print("The sum of ratios is equal to: {}%".format(sum_ratio))

print("############")
print('Number of pixels for class {}: {}'.format(3, num_classpixels[3]))

# Save textfile
PATH = '/home/exjobb/results/' + dataset + folder + 'PixelInfo.txt'
f = open(PATH, "w+")
f.write("DATASET PIXEL RATIOS \r\n")
f.write("##################################################### \r\n\r\n")
f.write("Dataset: %s \r\n" % dataset)
f.write("##################################################### \r\n\r\n")

for name, ratio in zip(class_names, class_ratios):
    f.write("Ratio of {:13} {:5.3f}% \r\n".format(name, ratio))

f.write("##################################################### \r\n\r\n")
f.write("The sum of ratios is equal to: {}%".format(sum_ratio))
f.close()

