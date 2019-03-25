import os
def create_textfile(Text_dir, Data_dir):
    # Save textfile
    f = open(Text_dir, "w+")
    img_list = os.listdir(Data_dir)
    for img in img_list:
        f.write("train/left/%s train/right/%s\r\n" % (img, img))
    f.close()


Text_dir = "/home/exjobb/DSegNet/pydnet/KITTI_stereo_train.txt"
Data_dir = "/MLDatasetsStorage/exjobb/KITTI/KITTI_stereo/train/left"
create_textfile(Text_dir, Data_dir)
