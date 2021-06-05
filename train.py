import tensorflow as tf
from tensorflow.keras import callbacks
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from .dataset import ADE20kDataset, PascalContextDataset
from .model import TGNet, GloReNet
import argparse
import os
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


parser = argparse.ArgumentParser()
parser.add_argument("-d","--dataset",default='pascal')
parser.add_argument("-m","--module",default='tgn')
parser.add_argument("-k","--backbone",default='effb5')
parser.add_argument("-b","--batchsize",default=8)
parser.add_argument("-s","--size",default=(320,320))
parser.add_argument("-e","--epochs",default=50)
parser.add_argument("-n","--node",default=128)
args = parser.parse_args()

class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    '''
    y_true: [n,h,w,1]
    y_pred: [n,h,w,num_classes]
    num_classes: for classfication
    '''
    def __init__(self,
                 y_true=None,
                 y_pred=None,
                 num_classes=None,
                 name='mIOU',
                 dtype=None):
        super(UpdatedMeanIoU, self).__init__(num_classes=num_classes,
                                             name=name,
                                             dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)


def scheduler(epoch, lr=0.001):
    if epoch < 3:
        lr = 0.0005
    elif epoch < 5:
        lr = 0.0004
    elif epoch < 10:
        lr = 0.0003
    elif epoch < 20:
        lr = 0.0002
    elif epoch < 40:
        lr = 0.0002
    else:
        lr = 0.0001
    return lr


if __name__ == '__main__':

    tf.print(tf.__version__)
    assert tf.config.list_physical_devices('GPU'), "Exit...No GPU device!"

    # get dataset
    if args.dataset == 'pascal':
        ds_train = PascalContextDataset(root='/home/junelin/data/PascalContext',
            size=args.size, split='train', batch_size=args.batchsize,data_augment=True) \
            .generate_dataset("/home/junelin/data/PascalContext/train.txt")
        ds_valid = PascalContextDataset(root='/home/junelin/data/PascalContext',
            size=args.size, split='validation', batch_size=1) \
            .generate_dataset("/home/junelin/data/PascalContext/validation.txt")
        num_class=60
    elif args.dataset == 'ade20k':
        ds_train = ADE20kDataset(root="/home/junelin/data/ADEChallengeData2016",
                                size=args.size,batch_size=args.batchsize, split='train',data_augment=False).generate_dataset()
        ds_valid = ADE20kDataset(root="/home/junelin/data/ADEChallengeData2016",
                                size=args.size,batch_size=1, split='validation').generate_dataset()
        num_class=151

    # get model
    mirrored_strategy = tf.distribute.MirroredStrategy()  # 分布式训练
    with mirrored_strategy.scope():
        if args.module == 'tgn':
            model = TGNet(backbone=args.backbone,input_shape=(*args.size,3),num_class=num_class,num_node=args.node)
        elif args.module == 'glore':
            model = GloReNet(backbone=args.backbone,input_shape=(*args.size,3),num_class=num_class,num_node=args.node)

        model.summary()
        # compile model
        radam = tfa.optimizers.RectifiedAdam(lr=1e-3)
        model.compile(optimizer=radam,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy',
                            UpdatedMeanIoU(num_classes=num_class)])

    # define result path
    date = time.strftime("%m%d_%H%M", time.localtime())
    result_dir = "/home/junelin/result/{}_{}_bs{}_{}*{}_{}" \
                 .format(date,args.dataset,args.batchsize,*args.size,args.backbone+args.module)
    
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
        print("The result is saved at %s\n"%result_dir)
        print("Using Model: {} \nNumber of Nodes in the Graph: {} \n".format(args.backbone+args.module,args.node))

    # define callbacks
    scheduler_callback = LearningRateScheduler(scheduler)   #考虑用余弦退火
    stopping_callback = EarlyStopping(monitor='mIOU',patience=5,min_delta=0.001,verbose=1,mode='max')
    checkpoint_callback = ModelCheckpoint('{}/val_miou_best.h5'.format(result_dir),
                                            monitor='val_mIOU',verbose=1,
                                            save_best_only=True,save_weights_only=False,mode='max')

    callbacks = [scheduler_callback,stopping_callback,checkpoint_callback]

    # start training
    model.fit(x=ds_train,epochs=args.epochs,callbacks=callbacks,validation_data=ds_valid
              ,verbose=2
              )
    model.save('{}/finished.h5'.format(result_dir))

    #finished training
    print('finished training!')