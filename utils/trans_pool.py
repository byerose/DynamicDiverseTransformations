# encoding = utf-8
# 变换的接口

import tensorflow as tf
import tensorflow.keras as keras

import numpy as np
import imgaug.augmenters as iaa
import skimage
import cv2
from scipy import ndimage

from utils.config import *
from utils.set_data import *

class Transforms(object):

    def geometric(self, original_images, transformation):
        """
        geometric transformations
        :param original_images:
        :param transformation:
        :return:
        """
        if MODE.DEBUG:
            print('Applying geometric transformation ({})...'.format(transformation))
        cha = 'last' # default channels mode
        if(len(original_images.shape)==3):
            if(original_images.shape[0] == 1 or original_images.shape[0] == 3):
                cha = 'first'
                original_images = set_channels_last(original_images)
            img_rows, img_cols, nb_channels = original_images.shape[:3]
        elif(len(original_images.shape)==4):
            if(original_images[0].shape[0] == 1 or original_images[0].shape[0] == 3):
                cha = 'first'
                original_images = set_channels_last(original_images)
            nb_images, img_rows, img_cols, nb_channels = original_images.shape[:4]    
        
        transformed_images = []
        
        if (transformation == TRANSFORMATION.geo_scale):
            if (nb_channels == 1):
                scale=0.9
            else:
                scale=1.2
            tform = skimage.transform.SimilarityTransform(scale=scale)
            if(len(original_images.shape)==3):
                transformed_images = skimage.transform.warp(original_images, tform)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(skimage.transform.warp(img, tform))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif (transformation == TRANSFORMATION.geo_perspective):
            aug = iaa.PerspectiveTransform(scale=(0.01, 0.15))
            original_images = skimage.util.img_as_ubyte(original_images)
            if(len(original_images.shape)==3):
                transformed_images = aug.augment_image(original_images)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(aug.augment_image(img))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
            transformed_images = transformed_images.astype(np.float32)
        elif (transformation == TRANSFORMATION.geo_piecewise):
            aug = iaa.PiecewiseAffine(scale=(0.01, 0.05))
            original_images = skimage.util.img_as_ubyte(original_images)
            if(len(original_images.shape)==3):
                transformed_images = aug.augment_image(original_images)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(aug.augment_image(img))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
            transformed_images = transformed_images.astype(np.float32)
        elif (transformation == TRANSFORMATION.geo_elastic):
            aug = iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25)
            original_images = skimage.util.img_as_ubyte(original_images)
            if(len(original_images.shape)==3):
                transformed_images = aug.augment_image(original_images)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(aug.augment_image(img))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
            transformed_images = transformed_images.astype(np.float32)        
        elif (transformation == TRANSFORMATION.geo_jigsaw):
            if(nb_channels==1):
                size = 4
            else:
                size = 5
            aug = iaa.Jigsaw(nb_rows=size, nb_cols=size)
            original_images = skimage.util.img_as_ubyte(original_images)
            if(len(original_images.shape)==3):
                transformed_images = aug.augment_image(original_images)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(aug.augment_image(img))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
            transformed_images = transformed_images.astype(np.float32)  
        elif (transformation == TRANSFORMATION.geo_swirl):
            strength = 3
            radius = 14
            if (nb_channels == 3):
                strength = 2
                radius = 16
            if(len(original_images.shape)==3):
                transformed_images = skimage.transform.swirl(original_images, strength=strength, radius=radius)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(skimage.transform.swirl(img, strength=strength, radius=radius))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))        
        else:
            raise ValueError('{} is not supported.'.format(transformation))
        transformed_images = (transformed_images-np.min(transformed_images))/(np.max(transformed_images)-np.min(transformed_images))
        if MODE.DEBUG:
            print('shapes: original - {}; transformed - {}'.format(original_images.shape, transformed_images.shape))
            print('Applied transformation {}.'.format(transformation))
        if(cha=='first'):
            return set_channels_first(transformed_images)
        else:
            return transformed_images


    def adjust(self, original_images, transformation):
        original_images = (original_images-np.min(original_images))/(np.max(original_images)-np.min(original_images))
        if MODE.DEBUG:
            print('Applying adjust transformation ({})...'.format(transformation))
        cha = 'last' # default channels mode
        if(len(original_images.shape)==3):
            if(original_images.shape[0] == 1 or original_images.shape[0] == 3):
                cha = 'first'
                original_images = set_channels_last(original_images)
            img_rows, img_cols, nb_channels = original_images.shape[:3]
        elif(len(original_images.shape)==4):
            if(original_images[0].shape[0] == 1 or original_images[0].shape[0] == 3):
                cha = 'first'
                original_images = set_channels_last(original_images)
            nb_images, img_rows, img_cols, nb_channels = original_images.shape[:4]

        if transformation == TRANSFORMATION.adjust_gamma:
            gamma = 2
            if(len(original_images.shape)==3):
                transformed_images = skimage.exposure.adjust_gamma(original_images, gamma=gamma)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(skimage.exposure.adjust_gamma(img, gamma=gamma))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif transformation == TRANSFORMATION.adjust_equalize_hist:
            nbins = 256
            if(len(original_images.shape)==3):
                transformed_images = skimage.exposure.equalize_adapthist(original_images, nbins=nbins)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(skimage.exposure.equalize_adapthist(img, nbins=nbins))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif transformation == TRANSFORMATION.adjust_log:
            if(len(original_images.shape)==3):
                transformed_images = skimage.exposure.adjust_log(original_images)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(skimage.exposure.adjust_log(img))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif transformation == TRANSFORMATION.adjust_sharpness:
            aug = iaa.pillike.EnhanceSharpness()
            original_images = skimage.util.img_as_ubyte(original_images)
            if(len(original_images.shape)==3):
                transformed_images = aug.augment_image(original_images)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(aug.augment_image(img))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
            transformed_images = transformed_images.astype(np.float32)
        elif transformation == TRANSFORMATION.adjust_brightness:
            aug = iaa.pillike.EnhanceBrightness()
            original_images = skimage.util.img_as_ubyte(original_images)
            if(len(original_images.shape)==3):
                transformed_images = aug.augment_image(original_images)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(aug.augment_image(img))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
            transformed_images = transformed_images.astype(np.float32)
        elif transformation == TRANSFORMATION.adjust_contrast:
            aug = iaa.pillike.EnhanceContrast()
            original_images = skimage.util.img_as_ubyte(original_images)
            if(len(original_images.shape)==3):
                transformed_images = aug.augment_image(original_images)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(aug.augment_image(img))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
            transformed_images = transformed_images.astype(np.float32)
        elif transformation == TRANSFORMATION.adjust_average_blur:
            aug = iaa.AverageBlur(k=2)
            original_images = skimage.util.img_as_ubyte(original_images)
            if(len(original_images.shape)==3):
                transformed_images = aug.augment_image(original_images)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(aug.augment_image(img))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
            transformed_images = transformed_images.astype(np.float32)
        elif transformation == TRANSFORMATION.adjust_gaussian_blur:
            aug = iaa.GaussianBlur(sigma=1.5)
            original_images = skimage.util.img_as_ubyte(original_images)
            if(len(original_images.shape)==3):
                transformed_images = aug.augment_image(original_images)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(aug.augment_image(img))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
            transformed_images = transformed_images.astype(np.float32)
        elif transformation == TRANSFORMATION.adjust_motion_blur:
            aug = iaa.MotionBlur(k=15)
            original_images = skimage.util.img_as_ubyte(original_images)
            if(len(original_images.shape)==3):
                transformed_images = aug.augment_image(original_images)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(aug.augment_image(img))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
            transformed_images = transformed_images.astype(np.float32)
        else:
            raise ValueError('{} is not supported.'.format(transformation))

        transformed_images = (transformed_images-np.min(transformed_images))/(np.max(transformed_images)-np.min(transformed_images))
        if MODE.DEBUG:
            print('shapes: original - {}; transformed - {}'.format(original_images.shape, transformed_images.shape))
            print('Applied transformation {}.'.format(transformation))
        if(cha=='first'):
            return set_channels_first(transformed_images)
        else:
            return transformed_images

    def compress(self, original_images, transformation):
        """

        :param original_images:
        :param transformation:
        :return:
        """
        images = original_images.copy()
        images *= 255.
        compress_rate = int(transformation.split('_')[-1])
        format = '.{}'.format(transformation.split('_')[1])

        cha = 'last' # default channels mode
        if(len(images.shape)==3):
            if(images.shape[0] == 1 or images.shape[0] == 3):
                cha = 'first'
                images = set_channels_last(images)
            img_rows, img_cols, nb_channels = images.shape[:3]
        elif(len(images.shape)==4):
            if(images[0].shape[0] == 1 or images[0].shape[0] == 3):
                cha = 'first'
                images = set_channels_last(images)
            nb_images, img_rows, img_cols, nb_channels = images.shape[:4]

        transformed_images = []
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_rate]

        if (format == '.png'):
            encode_param = [cv2.IMWRITE_PNG_COMPRESSION, compress_rate]

        if(len(images.shape)==3):
            result, encoded_img = cv2.imencode(format, images, encode_param)
            if False == result:
                print('Failed to encode image to jpeg format.')
                quit()
            # decode the image from encoded image
            decoded_img = cv2.imdecode(encoded_img, 1)
            if (nb_channels == 1):
                decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_RGB2GRAY)
            transformed_images = (decoded_img / 255.).astype(np.float32)
            if (nb_channels == 1):
                transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
        elif(len(images.shape)==4):
            transformed_images = []
            for img in images:
                result, encoded_img = cv2.imencode(format, img, encode_param)
                if False == result:
                    print('Failed to encode image to jpeg format.')
                    quit()
                # decode the image from encoded image
                decoded_img = cv2.imdecode(encoded_img, 1)
                if (nb_channels == 1):
                    decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_RGB2GRAY)
                transformed_images.append((decoded_img / 255.).astype(np.float32))
            transformed_images = np.stack(transformed_images, axis=0)
            if (nb_channels == 1):
                transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        transformed_images = (transformed_images-np.min(transformed_images))/(np.max(transformed_images)-np.min(transformed_images))
        if MODE.DEBUG:
            print('shapes: original - {}; transformed - {}'.format(images.shape, transformed_images.shape))
            print('Applied transformation {}.'.format(transformation))
        if(cha=='first'):
            return set_channels_first(transformed_images)
        else:
            return transformed_images

    def denoise(self, original_images, transformation):
        """
        denoising transformation
        :param original_images:
        :param transformation:
        :return:
        """
        cha = 'last' # default channels mode
        if(len(original_images.shape)==3):
            if(original_images.shape[0] == 1 or original_images.shape[0] == 3):
                cha = 'first'
                original_images = set_channels_last(original_images)
            img_rows, img_cols, nb_channels = original_images.shape[:3]
        elif(len(original_images.shape)==4):
            if(original_images[0].shape[0] == 1 or original_images[0].shape[0] == 3):
                cha = 'first'
                original_images = set_channels_last(original_images)
            nb_images, img_rows, img_cols, nb_channels = original_images.shape[:4]

        transformed_images = []
        channel_axis=2
        
        if (transformation == TRANSFORMATION.denoise_wavelet):
            method = 'VisuShrink'
            if (nb_channels == 3):
                method = 'BayesShrink'
            if(len(original_images.shape)==3):
                sigma_est = skimage.restoration.estimate_sigma(original_images, channel_axis=channel_axis, average_sigmas=True)
                transformed_images = skimage.restoration.denoise_wavelet(original_images, channel_axis=channel_axis, convert2ycbcr=False, method=method, mode='soft', sigma=sigma_est)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    sigma_est = skimage.restoration.estimate_sigma(img, channel_axis=channel_axis, average_sigmas=True)
                    transformed_images.append(skimage.restoration.denoise_wavelet(img, channel_axis=channel_axis, convert2ycbcr=False, method=method, mode='soft', sigma=sigma_est))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif (transformation == TRANSFORMATION.denoise_tv_chambolle):
            weight = 0.4
            if (nb_channels == 3):
                weight = 0.07
            if(len(original_images.shape)==3):
                transformed_images = skimage.restoration.denoise_tv_chambolle(original_images, weight=weight, channel_axis=channel_axis)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(skimage.restoration.denoise_tv_chambolle(img, weight=weight, channel_axis=channel_axis))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif (transformation == TRANSFORMATION.denoise_tv_bregman):
            eps = 1e-6
            max_num_iter = 50
            weight = 2
            if (nb_channels == 3):
                weight = 15
            if(len(original_images.shape)==3):
                transformed_images = skimage.restoration.denoise_tv_bregman(original_images, eps=eps, max_num_iter=max_num_iter, weight=weight)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(skimage.restoration.denoise_tv_bregman(img, eps=eps, max_num_iter=max_num_iter, weight=weight))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif (transformation == TRANSFORMATION.denoise_bilateral):
            if(len(original_images.shape)==3):
                if (nb_channels == 1):
                    original_images = np.squeeze(original_images)
                    channel_axis = None
                transformed_images = skimage.restoration.denoise_bilateral(original_images, sigma_color=0.05, sigma_spatial=15, channel_axis=channel_axis)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                if (nb_channels == 1):
                    original_images = np.squeeze(original_images)
                    channel_axis = None
                for img in original_images:
                    transformed_images.append(skimage.restoration.denoise_bilateral(img, sigma_color=0.05, sigma_spatial=15, channel_axis=channel_axis))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif (transformation == TRANSFORMATION.denoise_nl_means):
            patch_kw = dict(patch_size=5,  # 5x5 patches
                            patch_distance=6,  # 13x13 search area
                            )
            hr = 0.6
            sr = 1
            if (nb_channels == 3):
                sr = 3
            if(len(original_images.shape)==3):
                if (nb_channels == 1):
                    original_images = np.squeeze(original_images)
                    channel_axis = None            
                sigma_est = np.mean(skimage.restoration.estimate_sigma(original_images, channel_axis=channel_axis))
                transformed_images = skimage.restoration.denoise_nl_means(original_images, h=hr * sigma_est, sigma=sr * sigma_est, fast_mode=True, channel_axis=channel_axis, **patch_kw)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                if (nb_channels == 1):
                    original_images = np.squeeze(original_images)
                    channel_axis = None            
                for img in original_images:
                    sigma_est = np.mean(skimage.restoration.estimate_sigma(img, channel_axis=channel_axis))
                    transformed_images.append(skimage.restoration.denoise_nl_means(img, h=hr * sigma_est, sigma=sr * sigma_est, fast_mode=True,  channel_axis=channel_axis,**patch_kw))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        else:
            raise ValueError('{} is not supported.'.format(transformation))

        transformed_images = (transformed_images-np.min(transformed_images))/(np.max(transformed_images)-np.min(transformed_images))
        if MODE.DEBUG:
            print('shapes: original - {}; transformed - {}'.format(original_images.shape, transformed_images.shape))
            print('Applied transformation {}.'.format(transformation))
        if(cha=='first'):
            return set_channels_first(transformed_images)
        else:
            return transformed_images

    def noise(self, original_images, transformation):
        """
        Adding noise to given images.
        :param original_images:
        :param transformation:
        :return:
        """
        if MODE.DEBUG:
            print('Noising images({})...'.format(transformation))
        cha = 'last' # default channels mode
        if(len(original_images.shape)==3):
            if(original_images.shape[0] == 1 or original_images.shape[0] == 3):
                cha = 'first'
                original_images = set_channels_last(original_images)
            img_rows, img_cols, nb_channels = original_images.shape[:3]
        elif(len(original_images.shape)==4):
            if(original_images[0].shape[0] == 1 or original_images[0].shape[0] == 3):
                cha = 'first'
                original_images = set_channels_last(original_images)
            nb_images, img_rows, img_cols, nb_channels = original_images.shape[:4]

        if (transformation == TRANSFORMATION.noise_gaussian):
            if(len(original_images.shape)==3):
                transformed_images = skimage.util.random_noise(original_images, mode='gaussian', mean=0.01).astype(np.float32)
                if (nb_channels == 1):
                    # reshape a 3d array to a 4d array
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []

                for img in original_images:
                    transformed_images.append(skimage.util.random_noise(img, mode='gaussian', mean=0.01).astype(np.float32))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif (transformation == TRANSFORMATION.noise_localvar):
            if(len(original_images.shape)==3):
                transformed_images = skimage.util.random_noise(original_images, mode='localvar').astype(np.float32)
                if (nb_channels == 1):
                    # reshape a 3d array to a 4d array
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []

                for img in original_images:
                    transformed_images.append(skimage.util.random_noise(img, mode='localvar').astype(np.float32))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif (transformation == TRANSFORMATION.noise_poisson):
            if(len(original_images.shape)==3):
                transformed_images = skimage.util.random_noise(original_images, mode='poisson').astype(np.float32)
                if (nb_channels == 1):
                    # reshape a 3d array to a 4d array
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []

                for img in original_images:
                    transformed_images.append(skimage.util.random_noise(img, mode='poisson').astype(np.float32))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif (transformation == TRANSFORMATION.noise_salt):
            if(len(original_images.shape)==3):
                transformed_images = skimage.util.random_noise(original_images, mode='salt').astype(np.float32)
                if (nb_channels == 1):
                    # reshape a 3d array to a 4d array
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []

                for img in original_images:
                    transformed_images.append(skimage.util.random_noise(img, mode='salt').astype(np.float32))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif (transformation == TRANSFORMATION.noise_pepper):
            if(len(original_images.shape)==3):
                transformed_images = skimage.util.random_noise(original_images, mode='pepper').astype(np.float32)
                if (nb_channels == 1):
                    # reshape a 3d array to a 4d array
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []

                for img in original_images:
                    transformed_images.append(skimage.util.random_noise(img, mode='pepper').astype(np.float32))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif (transformation == TRANSFORMATION.noise_saltPepper):
            if(len(original_images.shape)==3):
                transformed_images = skimage.util.random_noise(original_images, mode='s&p').astype(np.float32)
                if (nb_channels == 1):
                    # reshape a 3d array to a 4d array
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []

                for img in original_images:
                    transformed_images.append(skimage.util.random_noise(img, mode='s&p').astype(np.float32))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif (transformation == TRANSFORMATION.noise_speckle):
            if(len(original_images.shape)==3):
                transformed_images = skimage.util.random_noise(original_images, mode='speckle', mean=0.01).astype(np.float32)
                if (nb_channels == 1):
                    # reshape a 3d array to a 4d array
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []

                for img in original_images:
                    transformed_images.append(skimage.util.random_noise(img, mode='speckle', mean=0.01).astype(np.float32))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        else:
            raise ValueError('{} is not supported.'.format(transformation))

        transformed_images = (transformed_images-np.min(transformed_images))/(np.max(transformed_images)-np.min(transformed_images))

        if MODE.DEBUG:
            print('shapes: original - {}; transformed - {}'.format(original_images.shape, transformed_images.shape))
            print('Applied transformation {}.'.format(transformation))
        if(cha=='first'):
            return set_channels_first(transformed_images)
        else:
            return transformed_images

    def filter(self, original_images, transformation):
        """
        :param original_images:
        :param transformation:
        :return:
        """
        if MODE.DEBUG:
            print('Applying filter transformation ({})...'.format(transformation))

        cha = 'last' # default channels mode
        if(len(original_images.shape)==3):
            if(original_images.shape[0] == 1 or original_images.shape[0] == 3):
                cha = 'first'
                original_images = set_channels_last(original_images)
            img_rows, img_cols, nb_channels = original_images.shape[:3]
        elif(len(original_images.shape)==4):
            if(original_images[0].shape[0] == 1 or original_images[0].shape[0] == 3):
                cha = 'first'
                original_images = set_channels_last(original_images)
            nb_images, img_rows, img_cols, nb_channels = original_images.shape[:4]

        transformed_images = []

        if (transformation == TRANSFORMATION.filter_sobel):
            if(len(original_images.shape)==3):
                transformed_images = skimage.filters.sobel(original_images)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(skimage.filters.sobel(img))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif (transformation == TRANSFORMATION.filter_median):
            if(len(original_images.shape)==3):
                transformed_images = ndimage.median_filter(original_images, size=3)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(ndimage.median_filter(img, size=3))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif (transformation == TRANSFORMATION.filter_minimum):
            if(len(original_images.shape)==3):
                transformed_images = ndimage.minimum_filter(original_images, size=3)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(ndimage.minimum_filter(img, size=3))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif (transformation == TRANSFORMATION.filter_maximum):
            if(len(original_images.shape)==3):
                transformed_images = ndimage.maximum_filter(original_images, size=3)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(ndimage.maximum_filter(img, size=3))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))        
        elif (transformation == TRANSFORMATION.filter_gaussian):
            if(len(original_images.shape)==3):
                transformed_images = ndimage.gaussian_filter(original_images, sigma=1)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(ndimage.gaussian_filter(img, sigma=1))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))            
        elif (transformation == TRANSFORMATION.filter_rank):
            if(len(original_images.shape)==3):
                transformed_images = ndimage.rank_filter(original_images, rank=15, size=3)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(ndimage.rank_filter(img, rank=15, size=3))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))                                 
        elif (transformation == TRANSFORMATION.filter_percentile):
            if(len(original_images.shape)==3):
                transformed_images = ndimage.percentile_filter(original_images, percentile=20, size=3)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(ndimage.percentile_filter(img, percentile=20, size=3))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))                                 
        else:
            raise ValueError('{} is not supported.'.format(transformation))

        transformed_images = (transformed_images-np.min(transformed_images))/(np.max(transformed_images)-np.min(transformed_images))

        if MODE.DEBUG:
            print('shapes: original - {}; transformed - {}'.format(original_images.shape, transformed_images.shape))
            print('Applied transformation {}.'.format(transformation))
        if(cha=='first'):
            return set_channels_first(transformed_images)
        else:
            return transformed_images

    def morphology(self, original_images, transformation):
        """
        Apply morphological transformations on images.
        :param: original_images - the images to applied transformations on.
        :param: transformation - the standard transformation to apply.
        :return: the transformed dataset.
        """
        if MODE.DEBUG:
            print('Applying morphological transformation ({})...'.format(transformation))

        cha = 'last' # default channels mode
        if(len(original_images.shape)==3):
            if(original_images.shape[0] == 1 or original_images.shape[0] == 3):
                cha = 'first'
                original_images = set_channels_last(original_images)
            img_rows, img_cols, nb_channels = original_images.shape[:3]
        elif(len(original_images.shape)==4):
            if(original_images[0].shape[0] == 1 or original_images[0].shape[0] == 3):
                cha = 'first'
                original_images = set_channels_last(original_images)
            nb_images, img_rows, img_cols, nb_channels = original_images.shape[:4]

        # set kernel as a matrix of size 2
        kernel = np.ones((2, 2),np.uint8)

        transformed_images = []

        if (transformation == TRANSFORMATION.morph_dilation):
            if(len(original_images.shape)==3):
                transformed_images = cv2.dilate(original_images, kernel, iterations=1)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(cv2.dilate(img, kernel, iterations=1))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif (transformation == TRANSFORMATION.morph_erosion):
            if(len(original_images.shape)==3):
                transformed_images = cv2.erode(original_images, kernel, iterations=1)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(cv2.erode(img, kernel, iterations=1))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif (transformation == TRANSFORMATION.morph_opening):
            if(len(original_images.shape)==3):
                transformed_images = cv2.morphologyEx(original_images, cv2.MORPH_OPEN, kernel)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif (transformation == TRANSFORMATION.morph_closing):
            if(len(original_images.shape)==3):
                transformed_images = cv2.morphologyEx(original_images, cv2.MORPH_CLOSE, kernel)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif (transformation == TRANSFORMATION.morph_gradient):
            if(len(original_images.shape)==3):
                transformed_images = cv2.morphologyEx(original_images, cv2.MORPH_GRADIENT, kernel)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif (transformation == TRANSFORMATION.morph_tophat):
            if(len(original_images.shape)==3):
                transformed_images = cv2.morphologyEx(original_images, cv2.MORPH_TOPHAT, kernel)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif (transformation == TRANSFORMATION.morph_blackhat):
            if(len(original_images.shape)==3):
                transformed_images = cv2.morphologyEx(original_images, cv2.MORPH_BLACKHAT, kernel)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))    
        else:
            raise ValueError('{} is not supported.'.format(transformation))

        transformed_images = (transformed_images-np.min(transformed_images))/(np.max(transformed_images)-np.min(transformed_images))

        if MODE.DEBUG:
            print('shapes: original - {}; transformed - {}'.format(original_images.shape, transformed_images.shape))
            print('Applied transformation {}.'.format(transformation))
        if(cha=='first'):
            return set_channels_first(transformed_images)
        else:
            return transformed_images

    def shift(self, original_images, transformation):
        """
        Shift images.
        :param: original_images - the images to applied transformations on.
        :param: transformation - the standard transformation to apply.
        :return: the transformed dataset.
        """
        if MODE.DEBUG:
            print('Shifting images({})...'.format(transformation))
        
        cha = 'last' # default channels mode
        if(len(original_images.shape)==3):
            if(original_images.shape[0] == 1 or original_images.shape[0] == 3):
                cha = 'first'
                original_images = set_channels_last(original_images)
            img_rows, img_cols, nb_channels = original_images.shape[:3]
        elif(len(original_images.shape)==4):
            if(original_images[0].shape[0] == 1 or original_images[0].shape[0] == 3):
                cha = 'first'
                original_images = set_channels_last(original_images)
            nb_images, img_rows, img_cols, nb_channels = original_images.shape[:4]
    
        tx = int(0.1 * img_cols)
        ty = int(0.1 * img_rows)

        if (transformation == TRANSFORMATION.shift_left):
            tx = 0 - tx
            ty = 0
        elif (transformation == TRANSFORMATION.shift_right):
            tx = tx
            ty = 0
        elif (transformation == TRANSFORMATION.shift_up):
            tx = 0
            ty = 0 - ty
        elif (transformation == TRANSFORMATION.shift_down):
            tx = 0
            ty = ty
        elif (transformation == TRANSFORMATION.shift_top_right):
            tx = tx
            ty = 0 - ty
        elif (transformation == TRANSFORMATION.shift_top_left):
            tx = 0 - tx
            ty = 0 - ty
        elif (transformation == TRANSFORMATION.shift_bottom_left):
            tx = 0 - tx
            ty = ty
        elif (transformation == TRANSFORMATION.shift_bottom_right):
            tx = tx
            ty = ty
        else:
            raise ValueError('{} is not supported.'.format(transformation))

        # define transformation matrix
        trans_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

        if(len(original_images.shape)==3):
            transformed_images = cv2.warpAffine(original_images, trans_matrix, (img_cols, img_rows))
            if (nb_channels == 1):
                # reshape a 3d array to a 4d array
                transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
        elif(len(original_images.shape)==4):
            # applying an affine transformation over the dataset
            transformed_images = []

            for img in original_images:
                transformed_images.append(cv2.warpAffine(img, trans_matrix, (img_cols, img_rows)))

            transformed_images = np.stack(transformed_images, axis=0)
            if (nb_channels == 1):
                # reshape a 3d array to a 4d array
                transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        transformed_images = (transformed_images-np.min(transformed_images))/(np.max(transformed_images)-np.min(transformed_images))
        if MODE.DEBUG:
            print('shapes: original - {}; transformed - {}'.format(original_images.shape, transformed_images.shape))
            print('Applied transformation {}.'.format(transformation))

        if(cha=='first'):
            return set_channels_first(transformed_images)
        else:
            return transformed_images
            
    def rotate(self, original_images, transformation):
        """
        Rotate images.
        :param: original_images - the images to applied transformations on.
        :param: transformation - the standard transformation to apply.
        :return: the transformed dataset.
        """
        if MODE.DEBUG:
            print('Rotating images({})...'.format(transformation))

        cha = 'last' # default channels mode
        if(len(original_images.shape)==3):
            if(original_images.shape[0] == 1 or original_images.shape[0] == 3):
                cha = 'first'
                original_images = set_channels_last(original_images)
            img_rows, img_cols, nb_channels = original_images.shape[:3]
        elif(len(original_images.shape)==4):
            if(original_images[0].shape[0] == 1 or original_images[0].shape[0] == 3):
                cha = 'first'
                original_images = set_channels_last(original_images)
            nb_images, img_rows, img_cols, nb_channels = original_images.shape[:4]
        center = (img_rows / 2, img_cols / 2)

        trans_matrix = None
        if (transformation == TRANSFORMATION.rotate_30):
            # rotate 30-deg counterclockwise
            angle = 30
            scale = 1.0

            trans_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        elif (transformation == TRANSFORMATION.rotate_60):
            # rotate 60-deg counterclockwise
            angle = 60
            scale = 1.0

            trans_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        elif (transformation == TRANSFORMATION.rotate_90):
            # rotate 90-deg counterclockwise
            angle = 90
            scale = 1.0

            trans_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        elif (transformation == TRANSFORMATION.rotate_180):
            # rotate 180-deg counterclockwise
            angle = 180
            scale = 1.0

            trans_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        elif (transformation == TRANSFORMATION.rotate_270):
            # rotate 270-deg counterclockwise
            angle = 270
            scale = 1.0

            trans_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        else:
            raise ValueError('{} is not supported.'.format(transformation))

        if(len(original_images.shape)==3):
            transformed_images = cv2.warpAffine(original_images, trans_matrix, (img_cols, img_rows))
            if (nb_channels == 1):
                # reshape a 3d array to a 4d array
                transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
        elif(len(original_images.shape)==4):
            # applying an affine transformation over the dataset
            transformed_images = []

            for img in original_images:
                transformed_images.append(cv2.warpAffine(img, trans_matrix, (img_cols, img_rows)))

            transformed_images = np.stack(transformed_images, axis=0)
            if (nb_channels == 1):
                # reshape a 3d array to a 4d array
                transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        transformed_images = (transformed_images-np.min(transformed_images))/(np.max(transformed_images)-np.min(transformed_images))
        if MODE.DEBUG:
            print('shapes: original - {}; transformed - {}'.format(original_images.shape, transformed_images.shape))
            print('Applied transformation {}.'.format(transformation))

        if(cha=='first'):
            return set_channels_first(transformed_images)
        else:
            return transformed_images

    def flip(self, original_images, transformation):
        """
        Flip images.
        :param: original_images - the images to applied transformations on.
        :param: transformation - the standard transformation to apply.
        :return: the transformed dataset.
        """
        if MODE.DEBUG:
            print('Flipping images({})...'.format(transformation))
        
        cha = 'last' # default channels mode
        if(len(original_images.shape)==3):
            if(original_images.shape[0] == 1 or original_images.shape[0] == 3):
                cha = 'first'
                original_images = set_channels_last(original_images)
            img_rows, img_cols, nb_channels = original_images.shape[:3]
        elif(len(original_images.shape)==4):
            if(original_images[0].shape[0] == 1 or original_images[0].shape[0] == 3):
                cha = 'first'
                original_images = set_channels_last(original_images)
            nb_images, img_rows, img_cols, nb_channels = original_images.shape[:4]

        # set flipping direction
        flip_direction = 0
        if (transformation == TRANSFORMATION.flip_vertical):
            # flip around the x-axis
            flip_direction = 0
        elif (transformation == TRANSFORMATION.flip_horizontal):
            # flip around the y-axis
            flip_direction = 1
        elif (transformation == TRANSFORMATION.flip_both):
            # flip around both axes
            flip_direction = -1
        else:
            raise ValueError('{} is not supported.'.format(transformation))

        if(len(original_images.shape)==3):
            transformed_images = cv2.flip(original_images, flip_direction)
            if (nb_channels == 1):
                # reshape a 3d array to a 4d array
                transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
        elif(len(original_images.shape)==4):
            # applying an affine transformation over the dataset
            transformed_images = []
            for img in original_images:
                transformed_images.append(cv2.flip(img, flip_direction))
            transformed_images = np.stack(transformed_images, axis=0)
            if (nb_channels == 1):
                # reshape a 3d array to a 4d array
                transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        if MODE.DEBUG:
            print('shapes: original - {}; transformed - {}'.format(original_images.shape, transformed_images.shape))
            print('Applied transformation {}.'.format(transformation))
        transformed_images = (transformed_images-np.min(transformed_images))/(np.max(transformed_images)-np.min(transformed_images))
        if(cha=='first'):
            return set_channels_first(transformed_images)
        else:
            return transformed_images

    def affine(self, original_images, transformation):
        """
        Apply affine transformation on images.
        :param: original_images - the images to applied transformations on.
        :param: transformation - the standard transformation to apply.
        :return: the transformed dataset.
        """
        if MODE.DEBUG:
            print('Applying affine transformation on images({})...'.format(transformation))

        """
        In affine transformation, all parallel lines in the original image will still be parallel in the transformed image.
        To find the transformation matrix, we need to specify 3 points from the original image 
        and their corresponding locations in transformed image. Then, the transformation matrix M (2x3) 
        can be generated by getAffineTransform()
        """
        cha = 'last' # default channels mode
        if(len(original_images.shape)==3):
            if(original_images.shape[0] == 1 or original_images.shape[0] == 3):
                cha = 'first'
                original_images = set_channels_last(original_images)
            img_rows, img_cols, nb_channels = original_images.shape[:3]
        elif(len(original_images.shape)==4):
            if(original_images[0].shape[0] == 1 or original_images[0].shape[0] == 3):
                cha = 'first'
                original_images = set_channels_last(original_images)
            nb_images, img_rows, img_cols, nb_channels = original_images.shape[:4]

        point1 = [0.25 * img_cols, 0.25 * img_rows]
        point2 = [0.25 * img_cols, 0.5 * img_rows]
        point3 = [0.5 * img_cols, 0.25 * img_rows]

        pts_original = np.float32([point1, point2, point3])

        if (transformation == TRANSFORMATION.affine_vertical_compress):
            point1 = [0.25 * img_cols, 0.32 * img_rows]
            point2 = [0.25 * img_cols, 0.48 * img_rows]
            point3 = [0.5 * img_cols, 0.32 * img_rows]
        elif (transformation == TRANSFORMATION.affine_vertical_stretch):
            point1 = [0.25 * img_cols, 0.2 * img_rows]
            point2 = [0.25 * img_cols, 0.55 * img_rows]
            point3 = [0.5 * img_cols, 0.2 * img_rows]
        elif (transformation == TRANSFORMATION.affine_horizontal_compress):
            point1 = [0.32 * img_cols, 0.25 * img_rows]
            point2 = [0.32 * img_cols, 0.5 * img_rows]
            point3 = [0.43 * img_cols, 0.25 * img_rows]
        elif (transformation == TRANSFORMATION.affine_horizontal_stretch):
            point1 = [0.2 * img_cols, 0.25 * img_rows]
            point2 = [0.2 * img_cols, 0.5 * img_rows]
            point3 = [0.55 * img_cols, 0.25 * img_rows]
        elif (transformation == TRANSFORMATION.affine_both_compress):
            point1 = [0.28 * img_cols, 0.28 * img_rows]
            point2 = [0.28 * img_cols, 0.47 * img_rows]
            point3 = [0.47 * img_cols, 0.28 * img_rows]
        elif (transformation == TRANSFORMATION.affine_both_stretch):
            point1 = [0.22 * img_cols, 0.22 * img_rows]
            point2 = [0.22 * img_cols, 0.55 * img_rows]
            point3 = [0.55 * img_cols, 0.22 * img_rows]
        else:
            raise ValueError('{} is not supported.'.format(transformation))

        # define transformation matrix
        pts_transformed = np.float32([point1, point2, point3])
        trans_matrix = cv2.getAffineTransform(pts_original, pts_transformed)

        if(len(original_images.shape)==3):
            transformed_images = cv2.warpAffine(original_images, trans_matrix, (img_cols, img_rows))
            if (nb_channels == 1):
                # reshape a 3d array to a 4d array
                transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
        elif(len(original_images.shape)==4):
            # applying an affine transformation over the dataset
            transformed_images = []
            for img in original_images:
                transformed_images.append(cv2.warpAffine(img, trans_matrix, (img_cols, img_rows)))
            transformed_images = np.stack(transformed_images, axis=0)
            if (nb_channels == 1):
                # reshape a 3d array to a 4d array
                transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        transformed_images = (transformed_images-np.min(transformed_images))/(np.max(transformed_images)-np.min(transformed_images))
        if MODE.DEBUG:
            print('shapes: original - {}; transformed - {}'.format(original_images.shape, transformed_images.shape))
            print('Applied transformation {}.'.format(transformation))
        if(cha=='first'):
            return set_channels_first(transformed_images)
        else:
            return transformed_images
