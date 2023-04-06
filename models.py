import keras

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU, Concatenate, GlobalAveragePooling2D, Input, BatchNormalization, SeparableConv2D, Subtract, concatenate
from keras.activations import relu, softmax
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras import backend as K
import mtcnn
import cv2
import numpy as np


def load_model():
    inputs = (3,225,225)
    squuezenet = SqueezeNet(128,inputs)
    squuezenet.load_weights("/home/saikrishna/Projects/face_id_dl/squeeze_net_model_final.h5")
    return  squuezenet


def extract_face(img, input_size, bbox):

  [X,Y,W,H] = bbox
  cropped_image = img[Y:Y+H, X:X+W]
  (c,x,y) = input_size
  resized_img = cv2.resize(cropped_image, (x,y),interpolation=cv2.INTER_LINEAR).astype(np.float32)
  resized_img = resized_img/255
  fixed_chanels = np.moveaxis(resized_img, 2, 0) 

  return fixed_chanels.reshape((1,3,225,225))


def euclidean_distance(inputs):
    assert len(inputs) == 2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    u, v = inputs
    return K.sqrt(K.sum((K.square(u - v)), axis=1, keepdims=True))
        

def get_mtcnn_bboxes(img):
    detector = mtcnn.MTCNN()
    faces = detector.detect_faces(img)
    bboxes = [face['box'] for face in faces] 
    return bboxes

def fire(x,squeeze,expand,name):


  fire_squeeze = Convolution2D(16, (1, 1), 
                                activation='relu', 
                                kernel_initializer='glorot_uniform',
        padding='same', name=name+"_squeeze",
        data_format="channels_first")(x)
  fire_expand1 = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name=name+'_expand1',
        data_format="channels_first")(fire_squeeze)
  fire_expand2 = Convolution2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name=name+'_expand2',
        data_format="channels_first")(fire_squeeze)
  merge = Concatenate(axis=1)([fire_expand1, fire_expand2])
  
  return merge
  

def SqueezeNet(nb_classes, inputs,name="SqueezeNet"):
    """ Keras Implementation of SqueezeNet(arXiv 1602.07360)
    @param nb_classes: total number of final categories
    Arguments:
    inputs -- shape of the input images (channel, cols, rows)
    """

    input_img = Input(shape=inputs)
    conv1 = Convolution2D(
        96, (7, 7), activation='relu', kernel_initializer='glorot_uniform',
        strides=(2, 2), padding='same', name='conv1',
        data_format="channels_first")(input_img)

    maxpool1 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool1',
        data_format="channels_first")(conv1)

    fire2 = fire(maxpool1,16,64,name='fire2')
    
    fire3 =  fire(fire2,16,64,name='fire3')

    fire4 =  fire(fire3,32,128,name='fire4')

    maxpool4 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool4',
        data_format="channels_first")(fire4)

    fire5 = fire(maxpool4,32,128,name='fire5')

    fire6 = fire(fire5,48,192,name='fire6')

    fire7 = fire(fire6,48,192,name='fire7')

    fire8 = fire(fire7,64,256,name='fire8')

    maxpool8 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool8',
        data_format="channels_first")(fire8)

    fire9 = fire(maxpool8,64,256,name='fire9')


    dropout = Dropout(0.5,)(fire9)
    conv10 = Convolution2D(
        nb_classes, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='valid', name='conv10',
        data_format="channels_first")(dropout)

    global_avgpool10 = GlobalAveragePooling2D(data_format='channels_first')(conv10)
    softmax = Activation("softmax", name='softmax')(global_avgpool10)

    return Model(inputs=input_img, outputs=softmax,name=name)