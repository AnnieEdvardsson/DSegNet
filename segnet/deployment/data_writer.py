import cv2
import os
import matplotlib as mpl
import matplotlib.cm as cm


def save_predictions(predictions, folder):
    for i in range(predictions.shape[0]):
        cv2.imwrite(folder + "/predictions/prediction" + str(i) + '.png', predictions[i])


def save_images_with_predictions(data, predictions, folder):
    for i in range(predictions.shape[0]):
        combined1 = predictions[i] * 0.6 + data[i, :, :, 0] * 0.7
        combined2 = predictions[i] * 0.6 + data[i, :, :, 1] * 0.7
        combined3 = predictions[i] * 0.6 + data[i, :, :, 2] * 0.7
        data[i, :, :, 0] = combined1
        data[i, :, :, 1] = combined2
        data[i, :, :, 2] = combined3
        cv2.imwrite(folder + "/combined/combined" + str(i) + '.png', data[i])

    # create_video(data, folder)


def create_video(data, video_folder):
    batch_size, height, width, _ = data.shape
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(os.path.join(video_folder, "video_predictions.avi"), fourcc, 2, (width, height))

    for i in range(batch_size):
        video.write(data[i])

    video.release()
    cv2.destroyAllWindows()
