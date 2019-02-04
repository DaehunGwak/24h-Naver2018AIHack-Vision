import threading

import numpy as np
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Add
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Reshape
from keras.layers import Dropout, BatchNormalization, Lambda, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50

def subblock(x, filter, **kwargs):
    x = BatchNormalization()()
    y = x
    y = Conv2D(filter, (1, 1), activation='relu', **kwargs)(y)
    y = BatchNormalization()(y)
    y = Conv2D(filter, (3, 3), activation='relu', **kwargs)(y)
    y = BatchNormalization()(y)
    y = Conv2D(K.int_shape(x)[-1], (1, 1), **kwargs)(y)
    y = Add()([x, y])
    y = Activation('relu')(y)
    return y

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


def get_siamese_model(input_shape=(224, 224, 3), embedding_dim=2048, weight_mode=None):
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
    # Best Model
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    x = Dense(embedding_dim, name='output_layer')(x)

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


def del_empty_class(image_gen, class_list, input_shape=(224, 224, 3), batch_size=32, path="."):
    result_list = class_list.copy()
    class_dict = dict()
    for i, c in enumerate(class_list):
        test_generator1 = image_gen.flow_from_directory(
            directory=path,
            subset='validation',
            classes=[c],
            target_size=input_shape[:2],
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42
        )
        class_dict[c] = len(test_generator1.filenames)
    for key, value in class_dict.items():
        if value == 0:
            result_list.remove(key)
    return result_list


def generate_samples(image_gen, flow_dir, batch_size=32, path=".", num_classes=1383, input_shape=(224, 224, 3),
                     class_mode=False, train_mode=None):
    class_list = list(flow_dir.class_indices.keys())
    if train_mode == 'validation':
        class_list = del_empty_class(image_gen, class_list, input_shape=input_shape, batch_size=batch_size,
                                     path=path)
    nb_classes = len(class_list)
    # print(nb_classes)
    now_class = -1
    while True:
        list_anchor = []
        list_positive = []
        list_negative = []
        pos_gen = None
        neg_gen = None

        for i in range(batch_size):
            # class_copy = class_list.copy()
            while True:
                if class_mode:
                    now_class = np.random.randint(nb_classes)
                else:
                    now_class = (now_class + 1) % nb_classes
                pos_gen = image_gen.flow_from_directory(
                    directory=path,
                    subset=train_mode,
                    classes=[class_list[now_class]],
                    target_size=input_shape[:2],
                    color_mode="rgb",
                    batch_size=2,
                    shuffle=True
                )
                if len(pos_gen.filenames) > 0:
                    break

            while True:
                neg_class = np.random.randint(nb_classes)
                while now_class == neg_class:
                    neg_class = np.random.randint(nb_classes)
                neg_gen = image_gen.flow_from_directory(
                    directory=path,
                    subset=train_mode,
                    classes=[class_list[neg_class]],
                    target_size=input_shape[:2],
                    color_mode="rgb",
                    batch_size=2,
                    shuffle=True
                )
                if len(neg_gen.filenames) > 0:
                    break

            pos_batch, _ = next(pos_gen)
            neg_batch, _ = next(neg_gen)
            list_anchor.append(pos_batch[0])
            list_positive.append(pos_batch[-1])
            list_negative.append(neg_batch[0])

        A = np.array(list_anchor)
        B = np.array(list_positive)
        C = np.array(list_negative)
        yield ({'anchor_input': A, 'positive_input': B, 'negative_input': C}, None)


''' the part of main eda
        print(train_generator.class_indices)
        class_list = list(train_generator.class_indices.keys())
        
        class_dict = dict()
        for i, c in enumerate(class_list):
            test_generator1 = train_datagen.flow_from_directory(
                directory=DATASET_PATH + '/train/train_data',
                subset='training',
                classes=[c],
                target_size=input_shape[:2],
                color_mode="rgb",
                batch_size=batch_size,
                class_mode="categorical",
                shuffle=True,
                seed=42
            )
            test_generator2 = train_datagen.flow_from_directory(
                directory=DATASET_PATH + '/train/train_data',
                subset='validation',
                classes=[c],
                target_size=input_shape[:2],
                color_mode="rgb",
                batch_size=batch_size,
                class_mode="categorical",
                shuffle=True,
                seed=42
            )
            class_dict[c] = (len(test_generator1.filenames), len(test_generator2.filenames))
            if i % 100 == 0:
                print("done", i, "\n")
        from pprint import pprint
        sorted_dict = sorted(class_dict.items(), key=lambda item: item[1][1])
        pprint(sorted_dict)
'''