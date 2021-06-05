import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph
from dataset import ADE20kDataset, PascalContextDataset


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


def get_flops(model):
    '''
    get the FLOPs of the input model
    FLOPs: floating point operations 浮点运算数 用于评估一个模型的运算复杂度
    '''
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops


if __name__ == '__main__':
    tf.print(tf.__version__)

    model = tf.keras.models.load_model('/home/junelin/result/0517_2252_pascal_bs8_320*320_effb5tgn/val_miou_best.h5',compile=False)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy',UpdatedMeanIoU(num_classes=151)])
    model.summary()
    print(get_flops(model))

    ds_train = PascalContextDataset(root='/home/junelin/data/PascalContext',
            size=(320,320), split='train', batch_size=1,data_augment=False) \
            .generate_dataset("/home/junelin/data/PascalContext/train.txt")
    ds_valid = PascalContextDataset(root='/home/junelin/data/PascalContext',
            size=(320,320), split='validation', batch_size=1,data_augment=False) \
            .generate_dataset("/home/junelin/data/PascalContext/validation.txt")

    # model.evaluate(ds_valid)
    model.evaluate(ds_train)