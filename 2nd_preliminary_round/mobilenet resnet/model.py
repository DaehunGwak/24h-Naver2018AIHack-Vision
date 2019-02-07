import numpy as np
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Add, Concatenate
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Reshape
from keras.layers import Dropout, BatchNormalization, Lambda, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50

g_embedding_model = None


def set_embedding_model(model):
    global g_embedding_model
    try:
        g_embedding_model = model
    except:
        return False
    return True


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


def get_siamese_model(input_shape=(224, 224, 3), embedding_dim=1024, weight_mode=None):
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
    input_layer = Input(shape=input_shape)
    mobile_base_model = MobileNet(weights=weight_mode, include_top=False,
                                  input_tensor=input_layer, input_shape=input_shape)
    resnet_base_model = ResNet50(weights=weight_mode, include_top=False,
                                 input_tensor=input_layer, input_shape=input_shape)

    for layer in resnet_base_model.layers:
        layer.name = layer.name + 'res'

    mobile_base_out = mobile_base_model.output
    resnet_base_out = resnet_base_model.output
    # Best Model
    mobile_base_out = GlobalAveragePooling2D()(mobile_base_out)
    mobile_base_out = Dropout(0.2)(mobile_base_out)
    mobile_base_out = BatchNormalization()(mobile_base_out)
    mobile_base_out = Dense(embedding_dim)(mobile_base_out)

    resnet_base_out = GlobalAveragePooling2D()(resnet_base_out)
    resnet_base_out = Dropout(0.2)(resnet_base_out)
    resnet_base_out = BatchNormalization()(resnet_base_out)
    resnet_base_out = Dense(embedding_dim)(resnet_base_out)

    x1 = Lambda(lambda x: x[0] * x[1])([mobile_base_out, resnet_base_out])
    x2 = Lambda(lambda x: x[0] + x[1])([mobile_base_out, resnet_base_out])
    x3 = Lambda(lambda x: K.abs(x[0] - x[1]))([mobile_base_out, resnet_base_out])
    x4 = Lambda(lambda x: K.square(x))(x3)
    x = Concatenate()([x1, x2, x3, x4])

    x = Lambda(lambda _x: K.l2_normalize(_x, axis=1))(x)
    embedding_model = Model(input_layer, x, name="embedding")

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
            batch_size=2,               # batch_size,
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
    num_sample_space = 1
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
            for _ in range(1, num_sample_space):
                pos_sample, _ = next(pos_gen)
                pos_batch = np.vstack((pos_batch, pos_sample))
            # print("pos_batch", pos_batch.shape)

            # generate negative samples
            neg_list = []
            for _ in range(num_sample_space):
                neg_class = np.random.randint(nb_classes)
                while now_class == neg_class or neg_class in neg_list:
                    neg_class = np.random.randint(nb_classes)
                neg_list.append(neg_class)
            neg_gen = gen_list[neg_list[0]]
            neg_batch, _ = next(neg_gen)
            for ni in range(1, num_sample_space):
                nc = gen_list[neg_list[ni]]
                neg_gen = gen_list[nc]
                neg_sample, _ = next(neg_gen)
                neg_batch = np.vstack((neg_batch, neg_sample))
            # print("neg_batch", neg_batch.shape)

            # pick up the anchor, positive, negative sample set
            list_anchor.append(pos_batch[0])
            list_positive.append(pos_batch[-1])
            list_negative.append(neg_batch[0])

        # casting to numpy
        A = np.array(list_anchor)
        B = np.array(list_positive)
        C = np.array(list_negative)
        yield ({'anchor_input': A, 'positive_input': B, 'negative_input': C}, None)
