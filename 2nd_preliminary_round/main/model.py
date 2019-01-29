import numpy as np
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Reshape, Dropout, BatchNormalization
from keras.applications.mobilenet import MobileNet


def get_model(input_shape=(224, 224, 3), num_classes=1383, weight_mode=None):
    """
    공인 CNN 불러오기
    :param input_shape:
    :param num_classes:
    :param weight_mode:
    :return:
    """
    base_model = MobileNet(weights=weight_mode, include_top=False, input_shape=input_shape)
    x = base_model.output
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    #  = Reshape([-1, 1, 1024])(x)
    x = Dropout(0.2)(x)
    # x = Conv2D(num_classes, kernel_size=[1, 1])(x)
    x = Dense(num_classes)(x)
    # x = Flatten()(x)
    x = Activation(activation='softmax', name='output_layer')(x)
    model = Model(base_model.input, x)
    return model



def get_siamese_model(input_shape=(224, 224, 3), embedding_dim=50, weight_mode=None):
    """
    샴 네트워크를 위해 임베딩 모델과, 트레플렛 학습용 모델을 얻을 수 있다.
    ## 용어정리
    * anchor(a) : query image
    * positive(p) : query 와 같은 분류인 image
    * negative(n) : query 와 다른 분류인 image
    :param input_shape: 인풋 shape을 지정, pretrained 모델 사용시 (224, 224, 3) 권장
    :param embedding_dim: feature extraction dimension 크기를 설정한다.
    :param weight_mode: 'imagenet'을 입력하면 pretrained 모델을 사용할 수 있다.
    :return: embedding_model, triplet_model(a, p, n)
    """
    base_model = MobileNet(weights=weight_mode, include_top=False, input_shape=input_shape)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(embedding_dim)(x)
    x = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)
    embedding_model = Model(base_model.input, x, name="embedding")

    anchor_input = Input(input_shape, name='anchor_input')
    positive_input = Input(input_shape, name='positive_input')
    negative_input = Input(input_shape, name='negative_input')
    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)

    inputs = [anchor_input, positive_input, negative_input]
    outputs = [anchor_embedding, positive_embedding, negative_embedding]

    triplet_model = Model(inputs, outputs)
    triplet_model.add_loss(K.mean(triplet_loss(outputs)))

    return embedding_model, triplet_model


def triplet_loss(inputs, dist='sqeuclidean', margin='maxplus'):
    """
    트리플렛 로스를 keras.backend 사용하여 제작
    :param inputs:
    :param dist:
    :param margin:
    :return:
    """
    anchor, positive, negative = inputs
    positive_distance = K.square(anchor - positive)
    negative_distance = K.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = K.sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = K.sum(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = K.maximum(0.0, 1 + loss)
    elif margin == 'softplus':
        loss = K.log(1 + K.exp(loss))
    return K.mean(loss)


def triplet_loss_np(inputs, dist='sqeuclidean', margin='maxplus'):
    """
    트리플렛 로스를 numpy를 사용하여 제작
    :param inputs:
    :param dist:
    :param margin:
    :return:
    """
    anchor, positive, negative = inputs
    positive_distance = np.square(anchor - positive)
    negative_distance = np.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = np.sqrt(np.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = np.sqrt(np.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = np.sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = np.sum(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = np.maximum(0.0, 1 + loss)
    elif margin == 'softplus':
        loss = np.log(1 + np.exp(loss))
    return np.mean(loss)