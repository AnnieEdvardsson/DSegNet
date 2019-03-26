"""
    Loss function for unsupervised learning for stereo image pairs. Include three different loss functions:
    - Appearance matching loss
    - Disparity smoothness loss
    - Left-Right Disparity Consistency Loss
"""

import numpy as np
import cv2
from skimage.measure import compare_ssim as ssim
from skimage import filters


class MonoLoss(object):
    def __init__(self, left_imgs, right_imgs, left_imgs_gray, right_imgs_gray):
        """

        :param left_imgs: [batch_size, height, width, colour_channel]
        :param right_imgs: [batch_size, height, width, colour_channel]
        """
        self.left_imgs = left_imgs
        self.right_imgs = right_imgs
        self.num_scale = 4
        self.alpha = 0.85

        w = 375
        h = 1242

        batch_size = 3

        self.scaled_leftimg = np.zeros(shape=(batch_size, w, h, 3, self.num_scale))
        self.scaled_leftimg = np.zeros(shape=(batch_size, w, h, 3, self.num_scale))
        self.disp_est_leftimg = np.zeros(shape=(batch_size, w, h, 1, self.num_scale))
        self.disp_est_leftimg = np.zeros(shape=(batch_size, w, h, 1, self.num_scale))

        print(self.scale_pyramid(left_imgs[0], 5).shape)

        self.scaled_leftimg[0] = self.scale_pyramid(left_imgs[0], 5)
        self.scaled_leftimg[1] = self.scale_pyramid(left_imgs[1], 5)
        self.scaled_leftimg[2] = self.scale_pyramid(left_imgs[2], 5)

        self.scaled_rightimg = [self.scale_pyramid(right_imgs[0], 5), self.scale_pyramid(right_imgs[1], 5),
                                self.scale_pyramid(right_imgs[2], 5)]

        self.left_est = [self.scale_pyramid(left_imgs_gray[0], 5), self.scale_pyramid(left_imgs_gray[1], 5),
                               self.scale_pyramid(left_imgs_gray[2], 5)]
        self.right_est = [self.scale_pyramid(right_imgs_gray[0], 5), self.scale_pyramid(right_imgs_gray[1], 5),
                                self.scale_pyramid(right_imgs_gray[2], 5)]

        #self.expand_disparitys()

        print(self.scaled_leftimg[1].shape)



    #def expand_disparitys(self):
        temp_leftimg = self.scale_pyramid(left_imgs[2], 5)
        temp_rightimg = self.scale_pyramid(right_imgs[2], 5)

        self.disp_est_leftimg = [np.expand_dims(disp, axis=3) for disp in temp_leftimg]
        self.disp_est_rightimg = [np.expand_dims(disp, axis=3) for disp in temp_rightimg]

        print('Total loss: {}'.format(self.build_losses()))

    def build_losses(self):
        Cap = self.appearance_matching_loss()
        Cds = self.disparity_smoothness_loss()
        Clr = self.left_right_disparity_consistency_loss()

        return Cap + Cds + Clr


    def scale_pyramid(self, img, num_scale):
        """

        :param img:
        :param num_scale:
        :return: scaled_imgs: [scale_size, batch_size, height, width, colour_channel]
        """
        scaled_imgs = []  # don't need full res images
        s = img.shape
        h = s[0]
        w = s[1]
        for i in range(num_scale - 1):
            ratio = 2 ** (i + 1)

            # if h % ratio is not 0 or w % ratio is not 0:
            #    raise ValueError("The image is not equally divided by {}".format(ratio))
            # REMOVE DUBBLE DIV LATER
            nh = h // ratio
            nw = w // ratio

            resized_img = cv2.resize(img, dsize=(nh, nw), interpolation=cv2.INTER_CUBIC)
            resized_img.astype(np.float32)

            scaled_imgs.append(resized_img)
        return scaled_imgs

    def appearance_matching_loss(self):
        ################################################################################################################
        # L1 - Least Absolute Deviations
        # Sum of the all the absolute differences between the true value and the predicted value.
        # Maximum difference = 255, Minimum difference = 0

        # Computes the absolute differences between the true value and the predicted value for each layer
        L1_leftimg = [np.abs(self.left_est[i] - self.scaled_leftimg[i]) for i in range(self.num_scale)]
        L1_rightimg = [np.abs(self.right_est[i] - self.scaled_rightimg[i]) for i in range(self.num_scale)]

        # Computes the mean L1 loss for each layer
        L1_loss_leftimg = [np.mean(i) for i in L1_leftimg]
        L1_loss_rightimg = [np.mean(i) for i in L1_rightimg]

        print('The L1 loss (left) = {}'.format(L1_loss_leftimg))

        ################################################################################################################
        # SSIM - Structural Similarity Index
        # Maximum difference = -1, Minimum difference = 1

        # Computes the mean SSIM for each layer
        SSIM_loss_leftimg = [ssim(self.left_est[i], self.scaled_leftimg[i], multichannel=True) for i in range(self.num_scale)]
        SSIM_loss_rightimg = [ssim(self.right_est[i], self.scaled_rightimg[i], multichannel=True) for i in range(self.num_scale)]

        print('The SSIM loss (left) = {}'.format(SSIM_loss_leftimg))
        ################################################################################################################
        # Weighted sum for appearance matching loss

        # CHECK HOW BIG SSIM FACTORN ARE (clip by value??)
        Cap_leftimg = [self.alpha * (1 - SSIM_loss_leftimg[i]) / 2 + (1 - self.alpha) * L1_loss_leftimg[i]
                       for i in range(self.num_scale)]

        Cap_rightimg = [self.alpha * (1 - SSIM_loss_rightimg[i]) / 2 + (1 - self.alpha) * L1_loss_rightimg[i]
                        for i in range(self.num_scale)]

        print('The Cap loss (left) = {}'.format(Cap_leftimg))

        Cap = Cap_leftimg + Cap_rightimg

        return Cap

    def disparity_smoothness_loss(self):
        disparity_smoothness_leftimg = self.get_disparity_smoothness(self.disp_est_leftimg, self.scaled_leftimg)
        disparity_smoothness_rightimg = self.get_disparity_smoothness(self.disp_est_rightimg, self.scaled_rightimg)

        Cds_leftimg = [np.mean(np.abs(disparity_smoothness_leftimg[i]))
                       / 2 ** i for i in range(self.num_scale)]
        Cds_rightimg = [np.mean(np.abs(disparity_smoothness_rightimg))
                        / 2 ** i for i in range(self.num_scale)]

        Cds = Cds_leftimg + Cds_rightimg

        return Cds

    def get_disparity_smoothness(self, disp_est, scaled_image):
        for disp, img in zip(disp_est, scaled_image):
            print(disp.shape)
            print(img.shape)


        # Compute the image gradient in x & y direction for each layer in the estimated disparity
        disp_gradients_x = [self.gradient_x(disp) for disp in disp_est]
        disp_gradients_y = [self.gradient_y(disp) for disp in disp_est]

        # Compute the image gradient in x & y direction for each layer in the scaled images
        image_gradients_x = [self.gradient_x(img) for img in scaled_image]
        image_gradients_y = [self.gradient_y(img) for img in scaled_image]

        for grad in image_gradients_x:
            print(grad.shape)

        # Compute the exponent term in the disparity smoothness loss for all layers
        weights_x = [np.exp(-np.mean(np.abs(grad), 3, keepdims=True)) for grad in image_gradients_x]
        weights_y = [np.exp(-np.mean(np.abs(grad), 3, keepdims=True)) for grad in image_gradients_y]

        # Multiply the exponent term with the disparity gradient and return the sum of x and y
        scales = 6 # <---- CHANGE THIIIIS?!?!
        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(scales)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(scales)]
        return smoothness_x + smoothness_y

    def gradient_x(self, img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(self, img):
        gy = img[:, :-1, :, :] - img[:, 1:, :, :]
        return gy


    def left_right_disparity_consistency_loss(self):
        # This cost attempts to make the left-view disparity map be equal to the projected right-view disparity map

        Clr_leftimg = [np.mean(np.abs(self.right_to_left_disp[i] - self.disp_left_est[i])) for i in
                           range(self.num_scale)]

        Clr_rightimg = [np.mean(np.abs(self.left_to_right_disp[i] - self.disp_right_est[i])) for i in
                            range(self.num_scale)]

        Clr = Clr_leftimg + Clr_rightimg

        return Clr