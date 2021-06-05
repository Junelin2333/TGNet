import tensorflow as tf
from tensorflow.keras.applications import ResNet101V2
from ade20k_dataset import ADE20kDataset
from pcontext_dataset import PascalContextDataset
import os
import cv2
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self,
                 y_true=None,
                 y_pred=None,
                 num_classes=None,
                 name='mean_iou',
                 dtype=None):
        super(UpdatedMeanIoU, self).__init__(num_classes=num_classes,
                                             name=name,
                                             dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

def create_visual_annotation(anno):
    """"""
    # assert np.max(anno) <= 7, "only 7 classes are supported, add new color in label2color_dict"
    label2color_array = np.asarray(
    [
      [0, 0, 0],
      [120, 120, 120],
      [180, 120, 120],
      [6, 230, 230],
      [80, 50, 50],
      [4, 200, 3],
      [120, 120, 80],
      [140, 140, 140],
      [204, 5, 255],
      [230, 230, 230],
      [4, 250, 7],
      [224, 5, 255],
      [235, 255, 7],
      [150, 5, 61],
      [120, 120, 70],
      [8, 255, 51],
      [255, 6, 82],
      [143, 255, 140],
      [204, 255, 4],
      [255, 51, 7],
      [204, 70, 3],
      [0, 102, 200],
      [61, 230, 250],
      [255, 6, 51],
      [11, 102, 255],
      [255, 7, 71],
      [255, 9, 224],
      [9, 7, 230],
      [220, 220, 220],
      [255, 9, 92],
      [112, 9, 255],
      [8, 255, 214],
      [7, 255, 224],
      [255, 184, 6],
      [10, 255, 71],
      [255, 41, 10],
      [7, 255, 255],
      [224, 255, 8],
      [102, 8, 255],
      [255, 61, 6],
      [255, 194, 7],
      [255, 122, 8],
      [0, 255, 20],
      [255, 8, 41],
      [255, 5, 153],
      [6, 51, 255],
      [235, 12, 255],
      [160, 150, 20],
      [0, 163, 255],
      [140, 140, 140],
      [250, 10, 15],
      [20, 255, 0],
      [31, 255, 0],
      [255, 31, 0],
      [255, 224, 0],
      [153, 255, 0],
      [0, 0, 255],
      [255, 71, 0],
      [0, 235, 255],
      [0, 173, 255],
      [31, 0, 255],
      [11, 200, 200],
      [255, 82, 0],
      [0, 255, 245],
      [0, 61, 255],
      [0, 255, 112],
      [0, 255, 133],
      [255, 0, 0],
      [255, 163, 0],
      [255, 102, 0],
      [194, 255, 0],
      [0, 143, 255],
      [51, 255, 0],
      [0, 82, 255],
      [0, 255, 41],
      [0, 255, 173],
      [10, 0, 255],
      [173, 255, 0],
      [0, 255, 153],
      [255, 92, 0],
      [255, 0, 255],
      [255, 0, 245],
      [255, 0, 102],
      [255, 173, 0],
      [255, 0, 20],
      [255, 184, 184],
      [0, 31, 255],
      [0, 255, 61],
      [0, 71, 255],
      [255, 0, 204],
      [0, 255, 194],
      [0, 255, 82],
      [0, 10, 255],
      [0, 112, 255],
      [51, 0, 255],
      [0, 194, 255],
      [0, 122, 255],
      [0, 255, 163],
      [255, 153, 0],
      [0, 255, 10],
      [255, 112, 0],
      [143, 255, 0],
      [82, 0, 255],
      [163, 255, 0],
      [255, 235, 0],
      [8, 184, 170],
      [133, 0, 255],
      [0, 255, 92],
      [184, 0, 255],
      [255, 0, 31],
      [0, 184, 255],
      [0, 214, 255],
      [255, 0, 112],
      [92, 255, 0],
      [0, 224, 255],
      [112, 224, 255],
      [70, 184, 160],
      [163, 0, 255],
      [153, 0, 255],
      [71, 255, 0],
      [255, 0, 163],
      [255, 204, 0],
      [255, 0, 143],
      [0, 255, 235],
      [133, 255, 0],
      [255, 0, 235],
      [245, 0, 255],
      [255, 0, 122],
      [255, 245, 0],
      [10, 190, 212],
      [214, 255, 0],
      [0, 204, 255],
      [20, 0, 255],
      [255, 255, 0],
      [0, 153, 255],
      [0, 41, 255],
      [0, 255, 204],
      [41, 0, 255],
      [41, 255, 0],
      [173, 0, 255],
      [0, 245, 255],
      [71, 0, 255],
      [122, 0, 255],
      [0, 255, 184],
      [0, 92, 255],
      [184, 255, 0],
      [0, 133, 255],
      [255, 214, 0],
      [25, 194, 194],
      [102, 255, 0],
      [92, 0, 255]
    ])

    # visualize
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):  # i for h
        for j in range(visual_anno.shape[1]):
            color = label2color_array[anno[i, j],:]
            color = color.reshape((3,))
            # print(color.shape,color)
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]

    return visual_anno



if __name__ == '__main__':
    size = (320,320)
    
    # ds_train = ADE20kDataset(size=(256, 320), batch_size=1).generate_dataset(
    #     root="E:\\Dataset\\ADEChallengeData2016", split='train')

    # ds_valid = ADE20kDataset(size=(256, 320), batch_size=1).generate_dataset(
    #     root="E:\\Dataset\\ADEChallengeData2016", split='validation')

    ds_train = PascalContextDataset(
        size=(320,320), split='validation', batch_size=1).generate_dataset(
            'E:\\Dataset\\PascalContext\\train.txt')
    ds_valid = PascalContextDataset(
        size=(320,320), split='validation', batch_size=1).generate_dataset(
            'E:\\Dataset\\PascalContext\\validation.txt')




    model = tf.keras.models.load_model(
        "D:\Python\\val_miou_best.h5",compile=False)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy',UpdatedMeanIoU(num_classes=151)])
    

    # model.summary()

    # print("\neffb0_glore result\n")
    # model.evaluate(ds_valid)
    # model.evaluate(ds_train)

    # exit()

    model2 = tf.keras.models.load_model(
        "D:\Python\毕业设计\\result\\0512_1251_pascal_bs8_320_320_res101tgn_node128\\val_miou_best.h5",compile=False)
    model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy',UpdatedMeanIoU(num_classes=151)])
    
    # model3 = tf.keras.models.load_model(
    #     "D:\Python\毕业设计\\result\\0513_1854_pascal_bs8_320_320_res101tgn_node256\\val_miou_best.h5",compile=False)
    # model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy',UpdatedMeanIoU(num_classes=151)])
    
    # model.summary()

    # print("\neffb0_glore result\n")
    # model.evaluate(ds_valid)
    # model.evaluate(ds_train)


    image = tf.io.read_file("E:\\Dataset\\PascalContext\\JPEGImages\\2008_000790.jpg")
    # image = tf.io.read_file('E:\Dataset\ADEChallengeData2016\images\\training\ADE_train_00017185.jpg')
    image = tf.io.decode_jpeg(image)
    image = tf.image.resize(image, size=size, method='bilinear')
    image = tf.cast(image, dtype='float32') / 255.0
    image = tf.reshape(image,[1,320,320,3])

    annotation = tf.io.read_file("E:\\Dataset\\PascalContext\\SegmentationClass\\2008_000790.png")
    # annotation = tf.io.read_file('E:\Dataset\ADEChallengeData2016\\annotations\\training\ADE_train_00017185.png')
    annotation = tf.io.decode_png(annotation)
    annotation = tf.image.resize(annotation, size=size, method='nearest')
    annotation = tf.cast(annotation, dtype='uint8')

    predict = model.predict(image,batch_size=1)
    predict = tf.argmax(predict,axis=-1)
    predict = tf.transpose(predict,perm=[1,2,0])
    predict = predict.numpy().astype('uint8')

    predict2 = model2.predict(image,batch_size=1)
    predict2 = tf.argmax(predict2,axis=-1)
    predict2 = tf.transpose(predict2,perm=[1,2,0])
    predict2 = predict2.numpy().astype('uint8')

    # predict3 = model3.predict(image,batch_size=1)
    # predict3 = tf.argmax(predict3,axis=-1)
    # predict3 = tf.transpose(predict3,perm=[1,2,0])
    # predict3 = predict3.numpy().astype('uint8')


    image = cv2.cvtColor(image.numpy().reshape([320,320,3]), cv2.COLOR_RGB2BGR)
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.imshow('annotation',create_visual_annotation(annotation.numpy()))
    cv2.waitKey(0)
    cv2.imshow('effb5',create_visual_annotation(predict))
    cv2.waitKey(0)
    cv2.imshow('res101',create_visual_annotation(predict2))
    cv2.waitKey(0)
    # cv2.imshow('tgn256',create_visual_annotation(predict3))
    # cv2.waitKey(0)

    # cv2.imwrite('image.jpg',image*255)
    # cv2.imwrite('annotation.png',create_visual_annotation(annotation.numpy()))
    # cv2.imwrite('glore.png',create_visual_annotation(predict))
    # cv2.imwrite('tgn_128.png',create_visual_annotation(predict2))
    # cv2.imwrite('tgn_256.png',create_visual_annotation(predict3))