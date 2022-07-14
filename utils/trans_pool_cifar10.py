# encoding = utf-8
import numpy as np
import imgaug.augmenters as iaa
import skimage
import cv2
from scipy import ndimage
from torch import rand

from utils.config_cifar10 import *
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
            aug = iaa.PerspectiveTransform(scale=(0.01, 0.05))
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
            aug = iaa.ElasticTransformation(alpha=(0.5, 5.0), sigma=2)
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
                strength = 1
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

        if transformation == TRANSFORMATION.adjust_sharpness:
            factor = (0.2,0.6)
            aug = iaa.pillike.EnhanceSharpness(factor=factor)
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
            factor = (1,2)
            aug = iaa.pillike.EnhanceBrightness(factor=factor)
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
            factor= (1,2)
            aug = iaa.pillike.EnhanceContrast(factor=factor)
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
        elif transformation == TRANSFORMATION.adjust_motion_blur:
            aug = iaa.MotionBlur(k=3)
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
                weight = np.random.uniform(0.05,0.08)
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
                weight = np.random.uniform(13,16)
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
            sigma_color = 0.05
            sigma_spatial = np.random.uniform(15,18)
            if(len(original_images.shape)==3):
                if (nb_channels == 1):
                    original_images = np.squeeze(original_images)
                    channel_axis = None
                transformed_images = skimage.restoration.denoise_bilateral(original_images, sigma_color=sigma_color, sigma_spatial=sigma_spatial, channel_axis=channel_axis)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                if (nb_channels == 1):
                    original_images = np.squeeze(original_images)
                    channel_axis = None
                for img in original_images:
                    transformed_images.append(skimage.restoration.denoise_bilateral(img, sigma_color=sigma_color, sigma_spatial=sigma_spatial, channel_axis=channel_axis))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif (transformation == TRANSFORMATION.denoise_nl_means):
            patch_kw = dict(patch_size=4,  # 5x5 patches
                            patch_distance=6,  # 13x13 search area
                            )
            hr = np.random.uniform(0.5,0.7)
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
            size = 2
            if(len(original_images.shape)==3):
                transformed_images = ndimage.median_filter(original_images, size=size)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(ndimage.median_filter(img, size=size))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif (transformation == TRANSFORMATION.filter_minimum):
            size = 2
            if(len(original_images.shape)==3):
                transformed_images = ndimage.minimum_filter(original_images, size=size)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(ndimage.minimum_filter(img, size=size))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif (transformation == TRANSFORMATION.filter_maximum):
            size = 2
            if(len(original_images.shape)==3):
                transformed_images = ndimage.maximum_filter(original_images, size=size)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(ndimage.maximum_filter(img, size=size))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))        
        elif (transformation == TRANSFORMATION.filter_gaussian):
            size = 2
            if(len(original_images.shape)==3):
                transformed_images = ndimage.uniform_filter(original_images, size=size)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(ndimage.uniform_filter(img, size=size))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))            
        elif (transformation == TRANSFORMATION.filter_rank):
            size = 2
            rank = np.random.randint(-1,2)
            if(len(original_images.shape)==3):
                transformed_images = ndimage.rank_filter(original_images, rank=rank, size=size)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(ndimage.rank_filter(img, rank=rank, size=size))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))                                 
        elif (transformation == TRANSFORMATION.filter_percentile):
            size = 2
            percentile = np.random.uniform(8,12)
            if(len(original_images.shape)==3):
                transformed_images = ndimage.percentile_filter(original_images, percentile=percentile, size=size)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(ndimage.percentile_filter(img, percentile=percentile, size=size))
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
    
        tx = 0
        ty = 0

        if (transformation == TRANSFORMATION.shift_left):
            tx = 0 - np.random.randint(2,5)
            ty = 0
        elif (transformation == TRANSFORMATION.shift_right):
            tx = np.random.randint(2,5)
            ty = 0
        elif (transformation == TRANSFORMATION.shift_up):
            tx = 0
            ty = 0 - 3
        elif (transformation == TRANSFORMATION.shift_down):
            tx = 0
            ty = np.random.randint(2,5)
        elif (transformation == TRANSFORMATION.shift_top_right):
            tx = np.random.randint(2,5)
            ty = 0 - np.random.randint(2,5)
        elif (transformation == TRANSFORMATION.shift_top_left):
            tx = 0 - np.random.randint(2,5)
            ty = 0 - np.random.randint(2,5)
        elif (transformation == TRANSFORMATION.shift_bottom_left):
            tx = 0 - np.random.randint(2,5)
            ty = np.random.randint(2,5)
        elif (transformation == TRANSFORMATION.shift_bottom_right):
            tx = np.random.randint(2,5)
            ty = np.random.randint(2,5)
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

        if (transformation == TRANSFORMATION.rotate_5):
            angle = np.random.uniform(3,7)
            if(len(original_images.shape)==3):
                transformed_images = skimage.transform.rotate(original_images, angle=angle)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(skimage.transform.rotate(img, angle=angle))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif (transformation == TRANSFORMATION.rotate_10):
            angle = np.random.uniform(8,12)
            if(len(original_images.shape)==3):
                transformed_images = skimage.transform.rotate(original_images, angle=angle)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(skimage.transform.rotate(img, angle=angle))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif (transformation == TRANSFORMATION.rotate_15):
            angle = np.random.uniform(13,17)
            if(len(original_images.shape)==3):
                transformed_images = skimage.transform.rotate(original_images, angle=angle)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                transformed_images = []
                for img in original_images:
                    transformed_images.append(skimage.transform.rotate(img, angle=angle))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    transformed_images = transformed_images.reshape((nb_images, img_rows, img_cols, nb_channels))
        elif (transformation == TRANSFORMATION.flip_horizontal):
            if(len(original_images.shape)==3):
                transformed_images = cv2.flip(original_images, 1)
                if (nb_channels == 1):
                    # reshape a 3d array to a 4d array
                    transformed_images = transformed_images.reshape((img_rows, img_cols, nb_channels))
            elif(len(original_images.shape)==4):
                # applying an affine transformation over the dataset
                transformed_images = []
                for img in original_images:
                    transformed_images.append(cv2.flip(img, 1))
                transformed_images = np.stack(transformed_images, axis=0)
                if (nb_channels == 1):
                    # reshape a 3d array to a 4d array
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

    