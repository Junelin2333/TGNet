from fcn import FCN8s
import tensorflow as tf
import os
import cv2
import numpy as np

import tensorflow.experimental.numpy as tnp


# ------------------------------------------------------------------------------
# GlobalParameters
SIZE = (384, 512)
BATCH_SIZE = 2 

classes = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor'
]
# Ground Truth with RGB color for each class
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
            [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
            [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
            [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

colormap = [(item[0] * 256 + item[1]) * 256 + item[2]
            for item in colormap]  #将RGB数组转换为十进制
label_matrix = np.zeros(256**3, dtype='int8')

# 标签映射  用数组空间去做一个映射，通过RGB十进制作为index进行访问，得到对应的label
for index, item in enumerate(colormap):
    label_matrix[item] = index

data_dir = "E:\\Dataset\\VOCdevkit\\VOC2012\\JPEGImages\\"
mask_dir = "E:\\Dataset\\VOCdevkit\\VOC2012\\SegmentationClass\\"

# ------------------------------------------------------------------------------

def mask_to_label(ground_truth):

    ground_truth = np.reshape(ground_truth, (-1, 3)).astype('int32')        # height*width*channels ->  n*channels

    label = np.zeros((384*512,1), dtype='int8')
 
    for i in range(label.shape[0]):
        index = (ground_truth[i,0]*256 + ground_truth[i,1])*256 + ground_truth[i,2]
        label[i] = label_matrix[index]
        
    label = tf.reshape(label,(384,512,1))
    return label


def load_image(img_path, size=SIZE, mode="data"):

    #读取图片
    image = tf.io.read_file(img_path)

    # image = tf.io.decode_image(image,expand_animations=False)
    # 上面的代码替换成下面的两组语句
    if mode == "data":
        image = tf.io.decode_jpeg(image)
    elif mode == "label":
        image = tf.io.decode_png(image)
    '''
    # tf.image.decode_png 等返回的 tensor 是有静态 shape的,
    # tf.image.decode_image 由于使用了 tf.cond 判断图片类型,因此返回的 tensor 没有静态 shape，
    # 造成它解码的图片无法与tf.image.resize_images() 一起使用.
    '''

    # if image.shape[1] == 500:      # width >= height
    #     pass
    # elif image.shape[0] == 500:
    #     image = tf.image.rot90(image)
    # if image.shape[1]>=image.shape[0]:
    #     pass
    # else:
    #     image = tf.image.rot90(image)

    '''
    对于annotation的缩放方式必须使用"nearest"
    因为annotation标记了每个像素的确切分类，是整数值
    如果使用"bilinear"缩放，会使缩放结果出现很多本来没有的值
    对image的缩放则应该使用"bilinear"或其他
    '''
    if mode == "data":
        image = tf.image.resize(image, size=size, method='bilinear')
        image = tf.cast(image, dtype='float32') / 255.0
        image = tf.reshape(image,(384,512,3))

    elif mode == "label":
        image = tf.image.resize(image, size=size, method='nearest')
        image = image.numpy()
        # image = tf.py_function(mask_to_label,inp=image,Tout=[tf.int8])
        image = mask_to_label(image)

    return image


def generate_dataset(pic_name):

    data_full_path = tf.strings.join([data_dir, pic_name, '.jpg'])
    '''
    注意  tf.py_function包装完之后返回值会多一个维度, 即把原来的返回值外面多包了一层壳
    
    '''
    # data = load_image(data_full_path, mode="data")
    data = tf.py_function(load_image,inp=[data_full_path,SIZE,"data"],Tout = [tf.float32])
    data = tf.squeeze(data,axis=0)

    mask_full_path = tf.strings.join([mask_dir, pic_name, '.png'])
    # label = load_image(mask_full_path, mode="label")
    label = tf.py_function(load_image,inp=[mask_full_path, SIZE, "label"],Tout=[tf.int8])
    label = tf.squeeze(label,axis=0)  #或者在mask_to_label用reshape,再用tf.squeeze删掉多生成的那个维度

    return (data,label)


if __name__ == "__main__":
    print(tf.__version__)
    
    train_data_dir = "E:\\Dataset\\VOCdevkit\\VOC2012\\ImageSets\\Segmentation\\train.txt"
    train_data = tf.data.TextLineDataset("E:\\Dataset\\VOCdevkit\\VOC2012\\ImageSets\\Segmentation\\train.txt")\
                        .map(generate_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                        .shuffle(buffer_size=32).batch(1) \
                        .prefetch(tf.data.experimental.AUTOTUNE)

    tf.print(train_data)

    # for data,label in train_data.take(1):
    #     tf.print(data.shape)  
    #     tf.print(label.shape) 
    #     cv2.imshow('data',data.numpy()[0,:,:])
    #     cv2.imshow('label',label.numpy()[0,:,:])
    #     cv2.waitKey(0)
    #     break

    # data, label = generate_dataset("2010_002532")
    # tf.print(data.shape)
    # tf.print(label.shape)

    model = FCN8s(21)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=[tf.keras.metrics.MeanIoU(num_classes=21)])
    model.fit(train_data,epochs=1)
