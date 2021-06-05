import os
import tensorflow as tf
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

class ADE20kDataset(object):

    def __init__(self,
                 root="/home/junelin/data/ADEChallengeData2016",
                 data_augment = False,
                 split='train',
                 size=(480,480),
                 num_classes=151,
                 batch_size=1):

        super(ADE20kDataset,self).__init__()

        self.root = root
        self.data_augment = data_augment
        self.size = size
        self.num_classes=num_classes
        self.split = split
        self.batch_size=batch_size

    def load_image_annotation(self, image_path):

        image = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(image, channels=3)

        annotation_path = tf.strings.regex_replace(image_path, 'images',
                                                   'annotations')
        annotation_path = tf.strings.regex_replace(annotation_path, 'jpg',
                                                   'png')  #掩膜是png格式
        annotation = tf.io.read_file(annotation_path)
        annotation = tf.io.decode_png(annotation, channels=1)
        # annotation = annotation - 1

        #  Data Augmentation
        if self.data_augment :
            image_annotation = tf.concat([image,annotation],axis=-1)

            image_annotation = tf.image.resize_with_crop_or_pad(image_annotation,512,512)
            image_annotation = tf.image.central_crop(image_annotation,0.75)
            image_annotation = tf.image.random_flip_left_right(image_annotation)

            image,annotation = image_annotation[:,:,:3],image_annotation[:,:,3:]
            # seq = iaa.Sequential([
            #         iaa.Fliplr(0.5),               # 50% flip horizon
            #         iaa.Sharpen((0.0, 1.0)),       # sharpen the image
            #         iaa.Affine(
            #             scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            #             translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            #             rotate=(-45, 45),
            #             shear=(-16, 16),
            #             order=[0, 1],
            #             cval=(0, 255),
            #             mode=ia.ALL
            #         )
            #     ], random_order=True)

            # annotation = SegmentationMapsOnImage(annotation.numpy(), shape=annotation.shape)
            # image,annotation = seq(image=image.numpy(), segmentation_maps=annotation)

        # 对于图像  使用双线性插值进行resize
        image = tf.image.resize(image, size=self.size, method='bilinear')
        image = tf.cast(image,dtype='float32') / 255.0
        # image = tf.image.per_image_standardization(image)   #这个api造成了很多的问题

        # 对于掩膜  使用最近邻插值进行resize
        annotation = tf.image.resize(annotation, size=self.size, method='nearest')
        annotation = tf.cast(annotation, dtype='uint8')
        annotation = tf.where(annotation == 255, np.dtype('uint8').type(0), annotation)

        return (image,annotation)
    

    def generate_dataset(self):

        assert os.path.exists(self.root), "The dataset directory is empty!"

        if self.split == 'train':
            image_dir = os.path.join(self.root, "images/training/*.*")

        elif self.split == 'validation':
            image_dir = os.path.join(self.root, "images/validation/*.*")

        data_pair = tf.data.Dataset.list_files(image_dir) \
                                    .map(self.load_image_annotation,
                                         num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                                    .shuffle(buffer_size=2000).batch(self.batch_size) \
                                    .prefetch(tf.data.experimental.AUTOTUNE)

        return data_pair


class PascalContextDataset(object):

    def __init__(self,
                 root="/home/junelin/data/PascalContext",
                 split='train',
                 size=(480,480),
                 num_classes=60,   #59+1
                 batch_size=8,
                 data_augment=True):

        super(PascalContextDataset,self).__init__()

        self.root = root
        self.size = size
        self.num_classes=num_classes
        self.split = split
        self.batch_size=batch_size
        self.data_augment = data_augment

    def load_image_annotation(self, file_name):

        image_path = tf.strings.join([self.root,'/JPEGImages/',file_name,'.jpg'])
        image = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(image, channels=3)

        annotation_path = tf.strings.join([self.root,'/SegmentationClass/',file_name,'.png'])
        annotation = tf.io.read_file(annotation_path)
        annotation = tf.io.decode_png(annotation, channels=1)

        if self.data_augment:
            image_annotation = tf.concat([image,annotation],axis=-1)

            image_annotation = tf.image.resize_with_crop_or_pad(image_annotation,512,512)
            image_annotation = tf.image.central_crop(image_annotation,0.75)
            image_annotation = tf.image.random_flip_left_right(image_annotation)
            
            image,annotation = image_annotation[:,:,:3],image_annotation[:,:,3:]
    

        # 对于图像  使用双线性插值进行resize
        image = tf.image.resize(image, size=self.size, method='bilinear')
        # image = tf.image.per_image_standardization(image)
        image = tf.cast(image,dtype='float32') / 255.0

        # 对于掩膜  使用最近邻插值进行resize
        annotation = tf.image.resize(annotation, size=self.size, method='nearest')
        annotation = tf.cast(annotation, dtype='uint8')
        annotation = tf.where(annotation == 255, np.dtype('uint8').type(0), annotation)

        return (image,annotation)
    

    def generate_dataset(self,file_path):

        assert os.path.exists(file_path), "The dataset directory is empty!"

        data_pair = tf.data.TextLineDataset(file_path) \
                                    .map(self.load_image_annotation,
                                         num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                                    .shuffle(buffer_size=2000).batch(self.batch_size) \
                                    .prefetch(tf.data.experimental.AUTOTUNE)

        return data_pair