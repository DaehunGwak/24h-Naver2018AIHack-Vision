# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import argparse
import pickle

import nsml
import numpy as np
import numpy.random as rng

from sklearn.model_selection import StratifiedKFold

from nsml import DATASET_PATH
import keras
from keras.callbacks import ReduceLROnPlateau
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenetv2 import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.layers import merge, Dense, Input, Lambda, GlobalMaxPooling2D, Flatten
from keras.models import  Model
from data_loader import train_data_loader

np.random.seed(1997)


def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(file_path):
        model.load_weights(file_path)
        print('model loaded!')

    def infer(queries, db):

        # Query 개수: 195
        # Reference(DB) 개수: 1,127
        # Total (query + reference): 1,322

        queries, query_img, references, reference_img = preprocess(queries, db)

        print('test data load queries {} query_img {} references {} reference_img {}'.
              format(len(queries), len(query_img), len(references), len(reference_img)))

        queries = np.asarray(queries)
        query_img = np.asarray(query_img)
        references = np.asarray(references)
        reference_img = np.asarray(reference_img)

        query_img = query_img.astype('float32')
        query_img /= 255
        reference_img = reference_img.astype('float32')
        reference_img /= 255

        """
        get_feature_layer = K.function([model.layers[0].input] + [K.learning_phase()], [model.layers[-1].output])

        print('inference start')

        # inference
        query_vecs = get_feature_layer([query_img, 0])[0]
        print("query shape:", query_vecs.shape)

        # caching db output, db inference
        db_output = './db_infer.pkl'
        if os.path.exists(db_output):
            with open(db_output, 'rb') as f:
                reference_vecs = pickle.load(f)
        else:
            reference_vecs = get_feature_layer([reference_img, 0])[0]
            with open(db_output, 'wb') as f:
                pickle.dump(reference_vecs, f)
        print("ref shape:", reference_vecs.shape)

        # l2 normalization
        query_vecs = l2_normalize(query_vecs)
        reference_vecs = l2_normalize(reference_vecs)
        
        # Calculate cosine similarity
        sim_matrix = np.dot(query_vecs, reference_vecs.T)
        """

        # Choose only one class by query class
        # sim_matrix = np.zeros((query_vecs.shape[0], reference_vecs.shape[0]))
        # for q in range(query_vecs.shape[0]):
        #     now_class = np.argmax(query_vecs[q])
        #     for r in range(reference_vecs.shape[0]):
        #         sim_matrix[q][r] = reference_vecs[r][now_class]

        """ predict """
        sim_matrix = np.zeros((query_img.shape[0], reference_img.shape[0]))
        # test predict
        input_shape = (224, 224, 3)
        x_shape = (1, input_shape[0], input_shape[1], input_shape[2])
        for iq, x_q in enumerate(query_img):
            x_qr = x_q.reshape(x_shape)
            for ir, x_r in enumerate(reference_img):
                x_rr = x_r.reshape(x_shape)
                sim_matrix[iq][ir] = model.predict([x_qr, x_rr])[0]

        """ parsing results """
        retrieval_results = {}


        for (i, query) in enumerate(queries):
            query = query.split('/')[-1].split('.')[0]
            sim_list = zip(references, sim_matrix[i].tolist())
            sorted_sim_list = sorted(sim_list, key=lambda x: x[1], reverse=True)

            ranked_list = [k.split('/')[-1].split('.')[0] for (k, v) in sorted_sim_list]  # ranked list

            retrieval_results[query] = ranked_list
        print('done')

        return list(zip(range(len(retrieval_results)), retrieval_results.items()))

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)


def l2_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# data preprocess
def preprocess(queries, db):
    query_img = []
    reference_img = []
    img_size = (224, 224)

    for img_path in queries:
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        query_img.append(img)

    for img_path in db:
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        reference_img.append(img)

    return queries, query_img, db, reference_img


def w_init(shape,name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)


def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)


def genPair(x, y, batch):
    left, limit = 0, x.shape[0]
    x_shape = (batch, x.shape[1], x.shape[2], x.shape[3])
    # print(x_shape)
    x_all_1 = np.zeros(x_shape)
    for it, x_one in enumerate(x):
        x_all_1[0:batch] = x_one
        left = 0
        while True:
            right = min(left + batch, limit)
            x_all_2 = x[left:right]
            Y = np.zeros((right - left, ))
            for y_i, ii in enumerate(range(left, right)):
                if (y[it] == y[ii]).all():
                    Y[y_i] = 1.0
                else:
                    Y[y_i] = 0.0
            # print([x_all_1.shape, x_all_2.shape], Y.shape)
            yield [x_all_1[:right - left], x_all_2], Y
            left = right
            if right == limit:
                break


if __name__ == '__main__':
    REDUCE_MODE = False     # reduce lr mode
    AUG_MODE = False        # augmentation mode
    PT_MODE = False         # pre trained(imagenet) mode

    TRAIN_STEP = 1000   # 100 step is full training 1 query
    VAL_STEP = 100

    EPOCHS = 10
    EPOCHS = 6100 // (TRAIN_STEP // 100)
    BATCH_SIZE = 64
    VAL_RATIO = 0.05
    FOLD_NUM = 8
    LR = 0.000025
    REDUCE_STEP = 10
    REDUCE_FACT = 0.25

    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epochs', type=int, default=EPOCHS)
    args.add_argument('--batch_size', type=int, default=BATCH_SIZE)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0', help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    config = args.parse_args()

    # training parameters
    nb_epoch = config.epochs
    batch_size = config.batch_size
    num_classes = 1000
    input_shape = (224, 224, 3)  # input image shape

    """ Model """
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    base = MobileNet(weights="imagenet", input_shape=input_shape, include_top=False)
    last = base.output
    x = GlobalMaxPooling2D()(last)
    base_model = Model(base.input, x)
    base_l = base_model(left_input)
    base_r = base_model(right_input)

    # Getting the L1 Distance between the 2 encodings
    L1_layer = Lambda(lambda tensor: K.abs(tensor[0] - tensor[1]))

    # Add the distance function to the network
    L1_distance = L1_layer([base_l, base_r])
    prediction = Dense(1, activation='sigmoid', bias_initializer=b_init)(L1_distance)
    model = Model(input=[left_input, right_input], output=prediction)

    '''
    if PT_MODE:
        model = MobileNetV2(weights='imagenet', input_shape=input_shape, classes=num_classes)
    else:
        model = MobileNetV2(weights=None, input_shape=input_shape, classes=num_classes)
    '''

    model.summary()
    bind_model(model)

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True

        """ Initiate Adam optimizer """
        opt = keras.optimizers.Adam(lr=LR)
        model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        """ Load data """
        print('dataset path', DATASET_PATH)
        output_path = ['./img_list.pkl', './label_list.pkl']
        train_dataset_path = DATASET_PATH + '/train/train_data'

        if nsml.IS_ON_NSML:
            # Caching file
            nsml.cache(train_data_loader, data_path=train_dataset_path, img_size=input_shape[:2],
                       output_path=output_path)
        else:
            # local에서 실험할경우 dataset의 local-path 를 입력해주세요.
            train_data_loader(train_dataset_path, input_shape[:2], output_path=output_path)

        with open(output_path[0], 'rb') as img_f:
            img_list = pickle.load(img_f)
        with open(output_path[1], 'rb') as label_f:
            label_list = pickle.load(label_f)

        x_all = np.asarray(img_list)
        labels = np.asarray(label_list)
        y_all = keras.utils.to_categorical(labels, num_classes=num_classes)
        x_all = x_all.astype('float32')
        x_all /= 255
        print(len(labels), 'train samples')

        from collections import defaultdict
        label_dict = defaultdict(lambda: 0)
        for yy in labels:
            label_dict[yy] += 1
        sorted_by_value = sorted(label_dict.items(), key=lambda kv: kv[1])
        for step in range(0 , 1000, 10):
            print(sorted_by_value[step:step+10])

        print("Generate data shape:", x_all.shape)

        # split train & validation
        # shuffle
        s = np.arange(x_all.shape[0])
        np.random.shuffle(s)
        x_all = x_all[s]
        y_all = y_all[s]
        labels = labels[s]
        # split
        split_i = int(x_all.shape[0] * (1.0 - VAL_RATIO))
        x_train = x_all[:split_i]
        y_train = y_all[:split_i]
        x_val = x_all[split_i:]
        y_val = y_all[split_i:]

        """ create generator """
        gen_train = genPair(x_train, y_train, BATCH_SIZE)
        gen_val = genPair(x_val, y_val, BATCH_SIZE)
        step_train = int(x_train.shape[0] * x_train.shape[0] / BATCH_SIZE)
        step_val = int(x_val.shape[0] * x_val.shape[0] / BATCH_SIZE)
        if (x_train.shape[0] * x_train.shape[0]) % BATCH_SIZE == 0:
            step_train += 1
        if (x_val.shape[0] * x_val.shape[0]) % BATCH_SIZE == 0:
            step_val += 1

        # K Fold Test
        '''
        kf = StratifiedKFold(n_splits=FOLD_NUM, shuffle=True)
        for t_i, v_i in kf.split(x_all, labels):
            x_train, x_val = x_all[t_i], x_all[v_i]
            y_train, y_val = y_all[t_i], y_all[v_i]
            break
        '''
        print("Train shape:", x_train.shape)
        print("Validation shape:", x_val.shape)

        """ Callback """
        monitor = 'loss'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3, mode='min', factor=0.5, min_lr=1e-10)

        """ Augmentaion """
        if AUG_MODE:
            datagen = ImageDataGenerator(zoom_range=0.3,
                                         shear_range=0.3,
                                         height_shift_range=0.1,
                                         width_shift_range=0.1,
                                         rotation_range=180,
                                         horizontal_flip=True,
                                         vertical_flip=True)


        """ Training loop """
        hist_all = []
        for epoch in range(nb_epoch):
            # Reduce LR : 샴에 맞게 리팩토링 해야함
            if REDUCE_MODE:
                if epoch % REDUCE_STEP == REDUCE_STEP - 1:
                    LR *= REDUCE_FACT
                    print("ReduceLR:", LR)
                    opt = keras.optimizers.Adam(lr=LR)
                    model.compile(loss='categorical_crossentropy',
                                  optimizer=opt,
                                  metrics=['accuracy'])


            # Train : 아직 어그 모드 개발해야함
            if AUG_MODE:
                res = model.fit_generator(generator=datagen.flow(x_train, y_train, batch_size=batch_size),
                                          steps_per_epoch=x_train.shape[0],
                                          initial_epoch=epoch,
                                          epochs=epoch + 1,
                                          callbacks=[reduce_lr],
                                          validation_data=(x_val, y_val),
                                          verbose=1,
                                          shuffle=True)
            else:
                res = model.fit_generator(generator=gen_train,
                                steps_per_epoch=TRAIN_STEP,
                                initial_epoch=epoch,
                                epochs=epoch + 1,
                                callbacks=[reduce_lr],
                                validation_data=gen_val,
                                validation_steps=VAL_STEP,
                                verbose=1,
                                shuffle=True)




            # save & print all logs
            hist_all.append(res.history)
            for i, hist in enumerate(hist_all):
                print(i, hist)
            train_loss, train_acc = res.history['loss'][0], res.history['acc'][0]
            val_loss, val_acc = res.history['val_loss'][0], res.history['val_acc'][0]

            # save model to nsml
            nsml.report(summary=True, epoch=epoch, epoch_total=nb_epoch,
                        loss=val_loss, acc=val_acc)
            while True:
                try:
                    nsml.save(epoch)
                except:
                    print("!!! NSML SAVE ERROR !!!, so retry ")
                    continue
                break


