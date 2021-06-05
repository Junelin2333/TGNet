import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Add,Concatenate,Conv1D,Activation,Softmax,BatchNormalization
from tensorflow.python.keras.layers.pooling import GlobalMaxPooling2D

def TGN(X,XL,N,C):
    '''
    Global Guide Graph Convolution Module
    '''
    _,h,w,_ = X.shape  # 或者用tf.shape

    # Projection ---------------------------------------------------------

    # 权重矩阵 B = θ(X;W_θ)
    B = Conv2D(N,1,(1,1))(X)                    # [n,h,w,N]
    B = tf.reshape(B,[-1,h*w,N])                # [n,h*w,N]
    B = tf.transpose(B,perm=[0,2,1])            # [n,h*w,N] -> [n,N,h*w]
    B_norm = Softmax(axis=2)(B)                 # 归一化

    #降维
    X_reduced = Conv2D(C,1,(1,1))(X)   
    Pool_Weights = GlobalMaxPooling2D()(X_reduced)
    Pool_Weights = tf.reshape(Pool_Weights,[-1,1,1,C])
    X_reduced = tf.reshape(X_reduced,[-1,h*w,C])           #[n,h*w,C]
    X_reduced = tf.transpose(X_reduced,perm=[0, 2, 1])   #[n,C,h*w]
    

    # 坐标空间 映射到 隐交互空间   其实就是codebook
    V = tf.matmul(X_reduced,tf.transpose(B_norm,perm=[0,2,1]))  # [n,C,h*w] * [n,h*w,N]= [n,C,N]    
    
    # Projection Done ----------------------------------------------------

    # Graph Convolution Unit ---------------------------------------------

    # 在Channels信息传播 所以卷的是Nodes [n,C,N]
    gcn = Conv1D(N,1,1,data_format='channels_last')(V)   
    #(I-Ag)V  
    gcn = V + gcn      
    # 在Nodes之间进行信息传播 所以卷的是Channels [n,C,N]
    gcn = Activation('relu')(gcn)
    gcn = Conv1D(C,1,1,data_format='channels_first')(gcn) 

    # GCN finished --------------------------------------------------------

    # ReProjection --------------------------------------------------------
    # 在这一步动手脚
    
    _,h_xl,w_xl,c_xl = XL.shape
    XL_reduced = Conv2D(C,1,(1,1))(XL)                       # [n,h2,w2,c2] -> [n,h2,w2,C]
    XL_reduced = XL_reduced + Pool_Weights                   # 加权重后的特征图os8
    XL_reduced = Conv2D(N,1,(1,1))(XL_reduced)               # [n,h2,w2,N]
    XL_reduced = tf.reshape(XL_reduced,[-1,h_xl*w_xl,N])     
    XL_reduced = tf.transpose(XL_reduced,perm=[0,2,1])       # [n,N,h2*w2]

    Y = tf.matmul(gcn,XL_reduced)           #  [n,C,N] * [n,N,h2*w2] = [n,C,h2*w2]
    Y = tf.transpose(Y,perm=[0,2,1])
    Y = tf.reshape(Y,[-1,h_xl,w_xl,C])      # [n,h2,w2,C] 

    output = Conv2D(c_xl,1,(1,1),activation='relu')(Y)
    output = BatchNormalization()(output)

    output = Add()([XL,output])     

    return output


def GloRe(X,N,C):
    _,h,w,c = X.shape
    B = Conv2D(N,1,1)(X)
    B = tf.reshape(B,[-1,N,h*w])

    X_reduced = Conv2D(C,1,1)(X)
    X_reduced = tf.reshape(X_reduced,[-1,C,h*w])

    V = tf.matmul(X_reduced,tf.transpose(B,perm=[0,2,1]))   #[n,C,N]
    
    gcn = Conv1D(N,1,1,data_format='channels_last')(V)
    gcn = gcn + V
    gcn = Activation('relu')(gcn)
    gcn = Conv1D(C,1,1,data_format='channels_first')(gcn)

    Y = tf.matmul(gcn,B)   #[n,C,N] * [n,N,h*w] = [n,C,h*w]
    Y = tf.reshape(Y,[-1,h,w,C])

    output = Conv2D(c,1,1,activation='relu')(Y)
    output = BatchNormalization()(output)
    output = Add()([output,X])

    return output
