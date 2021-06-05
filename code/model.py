from gcn import GloRe, TGN
import tensorflow as tf
from tensorflow.keras.applications import ResNet101V2, EfficientNetB5,EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate, BatchNormalization,UpSampling2D,Activation



def TGNet(num_class,input_shape,num_node,backbone='res101'):

    assert backbone == 'res101' or backbone == 'effb5', "Backbone must be 'res101' or 'effb5' !"

    if backbone == 'res101':
        backbone = ResNet101V2(include_top=False,weights='imagenet',input_shape=input_shape)
        os32 = backbone.get_layer('post_relu').output    # [None,10,10,2048]
        os16 = backbone.get_layer('conv4_block6_preact_relu').output # [None,20,20,1024]
        os8 = backbone.get_layer('conv3_block4_preact_relu').output # [None,40,40,512]
        os4 = backbone.get_layer('conv2_block3_preact_relu').output # [None,80,80,256]

        x = Conv2D(1024,1,activation = 'relu', padding = 'same',kernel_initializer ='he_normal')(os32)

        x = Conv2DTranspose(1024, 4, (2, 2), padding='same', activation='relu',kernel_initializer ='he_normal')(x)
        x = Concatenate(axis=-1)([os16, x])
        x = Conv2D(512,3,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = Conv2D(512,3,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = BatchNormalization()(x)

        
        x = Conv2DTranspose(512, 4, (2, 2), padding='same', activation='relu',kernel_initializer ='he_normal')(x)
        tgn = TGN(X=os32,XL=x,N=num_node,C=num_node*8)
        x = Concatenate(axis=-1)([os8, tgn])
        x = Conv2D(256,2,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = Conv2D(256,3,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(256, 4, (2, 2), padding='same', activation='relu',kernel_initializer ='he_normal')(x)
        x = Concatenate(axis=-1)([os4, x])
        x = Conv2D(256,3,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = Conv2D(256,3,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = BatchNormalization()(x)

        x = UpSampling2D((4,4),interpolation='bilinear')(x)
        x = Conv2D(256,2,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = Conv2D(256,3,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = BatchNormalization()(x)

        x = Conv2D(num_class,1,kernel_initializer ='he_normal')(x)

        outputs = Activation('softmax')(x)
        model = Model(inputs=backbone.input, outputs=outputs)

        return model

    
    if backbone == 'effb5':
        backbone = EfficientNetB5(include_top=False,weights='imagenet',input_shape=input_shape)
        os32 = backbone.get_layer(name='top_activation').output   #2048
        os16 = backbone.get_layer(name='block5g_add').output      #176
        os8 = backbone.get_layer(name='block3e_add').output       #64
        os4 = backbone.get_layer(name='block2b_add').output       #24

        # backbone = EfficientNetB0(include_top=False,weights='imagenet',input_shape=input_shape)
        # os32 = backbone.get_layer(name='top_activation').output       #[None,1/32,1/32,1280]
        # os16 = backbone.get_layer(name='block5c_add').output          #[None,1/16,1/16,112]
        # os8 = backbone.get_layer(name='block3b_add').output           #[None,1/8,1/8,40]
        # os4 = backbone.get_layer(name='block2b_add').output           #[None,1/4,1/4,24]

        x = Conv2D(1024,1,activation = 'relu', padding = 'same',kernel_initializer ='he_normal')(os32)

        x = Conv2DTranspose(256, 4, (2, 2), padding='same', activation='relu',kernel_initializer ='he_normal')(x)
        x = Concatenate(axis=-1)([os16, x])
        x = Conv2D(256,3,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = Conv2D(256,3,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = BatchNormalization()(x)


        x = Conv2DTranspose(128, 4, (2, 2), padding='same', activation='relu',kernel_initializer ='he_normal')(x)
        tgn = TGN(X=os32,XL=x,N=num_node,C=num_node*8)
        x = Concatenate(axis=-1)([os8, tgn])
        x = Conv2D(128,2,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = Conv2D(128,3,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(64, 4, (2, 2), padding='same', activation='relu',kernel_initializer ='he_normal')(x)
        x = Concatenate(axis=-1)([os4, x])
        x = Conv2D(64,3,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = Conv2D(64,3,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = BatchNormalization()(x)

        x = UpSampling2D((4,4),interpolation='bilinear')(x)
        x = Conv2D(64,2,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = Conv2D(64,3,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = BatchNormalization()(x)

        x = Conv2D(num_class,1,kernel_initializer ='he_normal')(x)

        outputs = Activation('softmax')(x)
        model = Model(inputs=backbone.input, outputs=outputs)

    return model

def GloReNet(num_class,input_shape,num_node,backbone='res101'):

    assert backbone == 'res101' or backbone == 'effb5', "Backbone must be 'res101' or 'effb5' !"

    if backbone == 'effb5':
        # backbone = EfficientNetB5(include_top=False,weights='imagenet',input_shape=input_shape)
        # os32 = backbone.get_layer(name='top_activation').output   #2048
        # os16 = backbone.get_layer(name='block5g_add').output      #176
        # os8 = backbone.get_layer(name='block3e_add').output       #64
        # os4 = backbone.get_layer(name='block2b_add').output       #24

        backbone = EfficientNetB0(include_top=False,weights='imagenet',input_shape=input_shape)
        os32 = backbone.get_layer(name='top_activation').output       #[None,1/32,1/32,1280]
        os16 = backbone.get_layer(name='block5c_add').output          #[None,1/16,1/16,112]
        os8 = backbone.get_layer(name='block3b_add').output           #[None,1/8,1/8,40]
        os4 = backbone.get_layer(name='block2b_add').output           #[None,1/4,1/4,24]

        x = Conv2D(1024,1,activation = 'relu', padding = 'same',kernel_initializer ='he_normal')(os32)
        glore = GloRe(X=os32,N=num_node,C=num_node*8)

        x = Conv2DTranspose(256, 4, (2, 2), padding='same', activation='relu',kernel_initializer ='he_normal')(glore)
        x = Concatenate(axis=-1)([os16, x])
        x = Conv2D(256,3,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = Conv2D(256,3,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = BatchNormalization()(x)


        x = Conv2DTranspose(128, 4, (2, 2), padding='same', activation='relu',kernel_initializer ='he_normal')(x)
        x = Concatenate(axis=-1)([os8, x])
        x = Conv2D(128,2,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = Conv2D(128,3,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(64, 4, (2, 2), padding='same', activation='relu',kernel_initializer ='he_normal')(x)
        x = Concatenate(axis=-1)([os4, x])
        x = Conv2D(64,3,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = Conv2D(64,3,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = BatchNormalization()(x)

        x = UpSampling2D((4,4),interpolation='bilinear')(x)
        x = Conv2D(64,2,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = Conv2D(64,3,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = BatchNormalization()(x)

        x = Conv2D(num_class,1,kernel_initializer ='he_normal')(x)

        outputs = Activation('softmax')(x)
        model = Model(inputs=backbone.input, outputs=outputs) 


    if backbone == 'res101':
        backbone = ResNet101V2(include_top=False,weights='imagenet',input_shape=input_shape)
        os32 = backbone.get_layer('post_relu').output    # [None,10,10,2048]
        os16 = backbone.get_layer('conv4_block6_preact_relu').output # [None,20,20,1024]
        os8 = backbone.get_layer('conv3_block4_preact_relu').output # [None,40,40,512]
        os4 = backbone.get_layer('conv2_block3_preact_relu').output # [None,80,80,256]

        x = Conv2D(1024,1,activation = 'relu', padding = 'same',kernel_initializer ='he_normal')(os32)
        glore = GloRe(X=os32,N=num_node,C=num_node*8)

        x = Conv2DTranspose(1024, 4, (2, 2), padding='same', activation='relu',kernel_initializer ='he_normal')(glore)
        x = Concatenate(axis=-1)([os16, x])
        x = Conv2D(512,2,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = Conv2D(512,3,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(512, 4, (2, 2), padding='same', activation='relu',kernel_initializer ='he_normal')(x)
        x = Concatenate(axis=-1)([os8, x])
        x = Conv2D(256,2,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = Conv2D(256,3,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(256, 4, (2, 2), padding='same', activation='relu',kernel_initializer ='he_normal')(x)
        x = Concatenate(axis=-1)([os4, x])
        x = Conv2D(256,3,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = Conv2D(256,3,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = BatchNormalization()(x)

        x = UpSampling2D((4,4),interpolation='bilinear')(x)
        x = Conv2D(256,2,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = Conv2D(256,3,activation = 'relu', padding = 'same', kernel_initializer ='he_normal')(x)
        x = BatchNormalization()(x)

        x = Conv2D(num_class,1,kernel_initializer ='he_normal')(x)

        outputs = Activation('softmax')(x)
        model = Model(inputs=backbone.input, outputs=outputs)
        
    return model