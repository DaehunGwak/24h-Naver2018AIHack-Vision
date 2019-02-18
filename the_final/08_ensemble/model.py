import threading
import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Add
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Reshape, GaussianNoise, AveragePooling2D
from keras.layers import Dropout, BatchNormalization, Lambda, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet169
from sklearn.metrics.pairwise import euclidean_distances
from keras.constraints import max_norm
from keras.initializers import Constant
from keras.layers import Layer
from classification_models.resnet import ResNet18
import keras

g_embedding_model = None


def set_embedding_model(model):
    global g_embedding_model
    try:
        g_embedding_model = model
    except:
        return False
    return True


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


def gem(x, p):
    _x = K.clip(x, min_value=1e-6, max_value=1e+6)
    _x = K.pow(x, p)
    _x = AveragePooling2D(strides=[x.shape[-2], x.shape[-1]])(_x)
    _x = K.pow(_x, 1. / p)
    return _x


def get_siamese_model(mobile_backbone_model, dense_backbone_model, input_shape=(224, 224, 3), embedding_dim=2048, p=3.):
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

    input_layer_metric = Input(shape=input_shape, name='metric_input')
    # mobile_base_model = MobileNet(weights=weight_mode, include_top=False,
    #                               input_tensor=input_layer, input_shape=input_shape)
    # dense_base_model = DenseNet169(weights=weight_mode, include_top=False,
    #                              input_tensor=input_layer, input_shape=input_shape)
    #
    # for layer in dense_base_model.layers:
    #     layer.name = layer.name + '_DenseNet'

    # mobile_base_out = mobile_backbone_model(inputs=input_layer)
    # dense_base_out = dense_backbone_model(inputs=input_layer)

    mobile_base_out = mobile_backbone_model(input_layer_metric)
    dense_base_out = dense_backbone_model(input_layer_metric)

    # mobile_base_out = mobile_base_model.output
    # dense_base_out = dense_base_model.output
    # Best Model
    # mobile_base_out = GlobalAveragePooling2D()(mobile_base_out)
    # mobile_base_out = Dropout(0.2)(mobile_base_out)
    # mobile_base_out = BatchNormalization()(mobile_base_out)
    mobile_base_out = Dense(embedding_dim, name='mobileNet')(mobile_base_out)

    # dense_base_out = GlobalAveragePooling2D()(dense_base_out)
    # dense_base_out = Dropout(0.2)(dense_base_out)
    # dense_base_out = BatchNormalization()(dense_base_out)
    dense_base_out = Dense(embedding_dim, name='denseNet')(dense_base_out)

    # x1 = Lambda(lambda x: x[0] * x[1])([mobile_base_out, dense_base_out])
    x = Lambda(lambda _x: _x[0] + _x[1], name='metric_lambda1')([mobile_base_out, dense_base_out])
    # x = Concatenate()([x1, x2])

    # base_model = DenseNet169(weights=weight_mode, include_top=False, input_shape=input_shape)
    #
    # x = base_model.output
    # x = BatchNormalization()(x)
    #
    # variable_p = K.variable(K.random_uniform(1, minval=0.9, maxval=1.1) * p)
    #
    # # Best Model
    # x = Lambda(gem, arguments={'p': variable_p})(x)
    # x = Flatten()(x)
    # x = GlobalAveragePooling2D()(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.2)(x)
    # x = Dense(embedding_dim, name='output_layer')(x)
    x = Lambda(lambda _x: K.l2_normalize(_x, axis=1), name='metric_lambda2')(x)
    # embedding_model = Model(base_model.input, x, name="embedding")
    embedding_model = Model(input_layer_metric, x, name="embedding")

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


def add_classification_dense_model(num_classes=1383, input_shape=(224, 224, 3), weight_mode='imagenet'):

    input_layer = Input(shape=input_shape, name='classification_input')
    mobile_base_model = MobileNet(weights=weight_mode, include_top=False,
                                  input_tensor=input_layer, input_shape=input_shape)
    dense_base_model = DenseNet169(weights=weight_mode, include_top=False,
                                   input_tensor=input_layer, input_shape=input_shape)

    for layer in dense_base_model.layers[1:]:
        layer.name = layer.name + '_DenseNet'

    for layer in mobile_base_model.layers[1:]:
        layer.name = layer.name + '_MobileNet'

    dense_base_output = dense_base_model.output
    dense_base_output = GlobalAveragePooling2D()(dense_base_output)
    dense_base_output = Dropout(0.2)(dense_base_output)
    dense_base_output = BatchNormalization()(dense_base_output)
    dense_backbone_model = Model(dense_base_model.input, dense_base_output, name='dense_backbone')
    predictions_dense = Dense(num_classes, activation='softmax')(dense_base_output)

    mobile_base_output = mobile_base_model.output
    mobile_base_output = GlobalAveragePooling2D()(mobile_base_output)
    mobile_base_output = Dropout(0.2)(mobile_base_output)
    mobile_base_output = BatchNormalization()(mobile_base_output)
    mobile_backbone_model = Model(mobile_base_model.input, mobile_base_output, name='mobile_backbone')
    predictions_mobile = Dense(num_classes, activation='softmax')(mobile_base_output)

    predictions = Lambda(lambda x: (x[0] + x[1]) / 2, name='classification_lambda')([predictions_dense, predictions_mobile])

    model = Model(inputs=input_layer, outputs=predictions)
    return mobile_backbone_model, dense_backbone_model, model


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


def get_all_generator(image_gen, class_list, input_shape=(224, 224, 3), batch_size=32, path=".", train_mode=None):
    result_list = []
    generator_list = []
    for i, c in enumerate(class_list):
        temp_list = class_list.copy()
        temp_list.remove(c)
        now_generator = image_gen.flow_from_directory(
            directory=path,
            subset=train_mode,
            classes=[c],
            target_size=input_shape[:2],
            color_mode="rgb",
            batch_size=8,               # batch_size,
            class_mode="binary",
            shuffle=True,
            seed=42
        )
        num_file = len(now_generator.filenames)
        if num_file > 0:
            result_list.append(c)
            generator_list.append(now_generator)
    return result_list, generator_list


def generate_samples(image_gen, flow_dir, batch_size=32, path=".", input_shape=(224, 224, 3),
                     random_mode=False, train_mode=None):
    global g_embedding_model
    num_neg_class = 4
    class_list = list(flow_dir.class_indices.keys())
    class_list, gen_list = get_all_generator(image_gen, class_list,
                                             input_shape=input_shape, batch_size=batch_size,
                                             path=path, train_mode=train_mode)
    nb_classes = len(class_list)
    now_class = -1
    while True:
        list_anchor = []
        list_positive = []
        list_negative = []
        pos_gen = None
        neg_gen = None

        for i in range(batch_size):
            # generate positive samples
            if random_mode:
                now_class = np.random.randint(nb_classes)
            else:
                now_class = (now_class + 1) % nb_classes
            pos_gen = gen_list[now_class]
            pos_batch, _ = next(pos_gen)

            # generate negative samples
            neg_list = []
            for _ in range(num_neg_class):
                neg_class = np.random.randint(nb_classes)
                while now_class == neg_class or neg_class in neg_list:
                    neg_class = np.random.randint(nb_classes)
                neg_list.append(neg_class)
            neg_gen = gen_list[neg_list[0]]
            neg_batch, _ = next(neg_gen)
            for ni in range(1, num_neg_class):
                neg_gen = gen_list[neg_list[ni]]
                neg_sample, _ = next(neg_gen)
                neg_batch = np.vstack((neg_batch, neg_sample))

            # distance
            pos_vec = g_embedding_model.predict(pos_batch)
            neg_vec = g_embedding_model.predict(neg_batch)
            anchor_vec = pos_vec[0].reshape(1, -1)
            pos_dis = euclidean_distances(anchor_vec, pos_vec)
            neg_dis = euclidean_distances(anchor_vec, neg_vec)
            pos_argmax = np.argmax(pos_dis)
            neg_argmin = np.argmin(neg_dis)

            # pick up the anchor, positive, negative sample set
            list_anchor.append(pos_batch[0])
            list_positive.append(pos_batch[pos_argmax])
            list_negative.append(neg_batch[neg_argmin])

        # casting to numpy
        A = np.array(list_anchor)
        B = np.array(list_positive)
        C = np.array(list_negative)
        yield ({'anchor_input': A, 'positive_input': B, 'negative_input': C}, None)