import os
import re
from matplotlib import pyplot as plt 
import numpy as np




if __name__ == "__main__":

    res101 = open(
        "D:\Python\毕业设计\\result\\0512_1251_pascal_bs8_320_320_res101tgn_node128\\nohup.out",
        'r')
    effb0 = open(
        "D:\Python\毕业设计\\result\\0510_1056_pascal_bs8_320_320_effb0tgn\\nohup.out",
        'r')
    effb5 = open(
        "D:\Python\毕业设计\\result\\0518_2228_pascal_bs8_320_320_effb5tgn_node128\\nohup.out",
        'r')

    loss = []
    acc = []
    miou = []
    val_loss = []
    val_acc = []
    val_miou = []


    for line in res101:

        if re.search(r'(.*)- loss:(.*)val_mIOU: (.*)',line):

            loss.append(re.findall(r'- loss: (.*) - accuracy',line)[0])
            acc.append(re.findall(r'- accuracy: (.*) - mIOU',line)[0])
            miou.append(re.findall(r'- mIOU: (.*) - val_loss',line)[0])
            val_loss.append(re.findall(r'- val_loss: (.*) - val_accuracy',line)[0])
            val_acc.append(re.findall(r'- val_accuracy: (.*) - val_mIOU',line)[0])
            val_miou.append(re.findall(r'- val_mIOU: (.*)',line)[0])

    loss1 = list(map(float,loss))
    acc1 = list(map(float,acc))
    miou1 = list(map(float,miou))
    val_loss1 = list(map(float,val_loss))
    val_acc1 = list(map(float,val_acc))
    val_miou1 = list(map(float,val_miou))

    loss = []
    acc = []
    miou = []
    val_loss = []
    val_acc = []
    val_miou = []
    
    for line in effb0:

        if re.search(r'(.*)- loss:(.*)val_mIOU: (.*)',line):

            loss.append(re.findall(r'- loss: (.*) - accuracy',line)[0])
            acc.append(re.findall(r'- accuracy: (.*) - mIOU',line)[0])
            miou.append(re.findall(r'- mIOU: (.*) - val_loss',line)[0])
            val_loss.append(re.findall(r'- val_loss: (.*) - val_accuracy',line)[0])
            val_acc.append(re.findall(r'- val_accuracy: (.*) - val_mIOU',line)[0])
            val_miou.append(re.findall(r'- val_mIOU: (.*)',line)[0])

    loss2 = list(map(float,loss))
    acc2 = list(map(float,acc))
    miou2 = list(map(float,miou))
    val_loss2 = list(map(float,val_loss))
    val_acc2 = list(map(float,val_acc))
    val_miou2 = list(map(float,val_miou))

    loss = []
    acc = []
    miou = []
    val_loss = []
    val_acc = []
    val_miou = []

    for line in effb5:

        if re.search(r'(.*)- loss:(.*)val_mIOU: (.*)',line):

            loss.append(re.findall(r'- loss: (.*) - accuracy',line)[0])
            acc.append(re.findall(r'- accuracy: (.*) - mIOU',line)[0])
            miou.append(re.findall(r'- mIOU: (.*) - val_loss',line)[0])
            val_loss.append(re.findall(r'- val_loss: (.*) - val_accuracy',line)[0])
            val_acc.append(re.findall(r'- val_accuracy: (.*) - val_mIOU',line)[0])
            val_miou.append(re.findall(r'- val_mIOU: (.*)',line)[0])

    loss3 = list(map(float,loss))
    acc3 = list(map(float,acc))
    miou3 = list(map(float,miou))
    val_loss3 = list(map(float,val_loss))
    val_acc3 = list(map(float,val_acc))
    val_miou3 = list(map(float,val_miou))

    epoch = np.arange(1,len(loss)+1)

    plt.figure()

    plt.subplot(1,2,1)
    plt.title('MeanIOU on training dataset')
    plt.xlabel('Epoch')
    plt.plot(epoch,miou1,'r',label='ResNet-101')
    plt.plot(epoch,miou2,'g',label='EfficientNet-B0')
    plt.plot(epoch,miou3,'b',label='EfficientNet-B5')
    plt.legend()

    plt.subplot(1,2,2)
    plt.title('MeanIOU on validation dataset')
    plt.xlabel('Epoch')
    plt.plot(epoch,val_miou1,'r',label='ResNet-101')
    plt.plot(epoch,val_miou2,'g',label='EfficientNet-B0')
    plt.plot(epoch,val_miou3,'b',label='EfficientNet-B5')
    plt.legend()



    plt.show()