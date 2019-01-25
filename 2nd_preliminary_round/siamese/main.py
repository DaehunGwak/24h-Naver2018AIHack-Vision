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

from nsml import DATASET_PATH
import keras
from keras.callbacks import ReduceLROnPlateau
from keras.applications.mobilenetv2 import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, Flatten
from keras.models import Model
from keras import backend as K
from data_loader import train_data_loader
import model as MODEL

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
        input_shape = (224, 224, 3)

        """ create generator """
        datagen = ImageDataGenerator(rescale=1. / 255.)
        query_gen = datagen.flow_from_directory('/'.join(queries[0].split("/")[:-2]),
                                                classes=['query'], target_size=input_shape[:2])
        ref_gen = datagen.flow_from_directory('/'.join(db[0].split("/")[:-2]),
                                              classes=['reference'], target_size=input_shape[:2])

        """ type cast to numpy """
        queries = np.asarray(queries)
        references = np.asarray(db)

        """ inference """
        print('inference start')
        query_vecs = model.predict_generator(query_gen, steps=len(query_gen))
        print("query shape:", query_vecs.shape)
        reference_vecs = model.predict_generator(ref_gen, steps=len(ref_gen))
        print("ref shape:", reference_vecs.shape)

        """ Calculate cosine similarity """
        sim_matrix = np.dot(query_vecs, reference_vecs.T)

        """ Parsing Results """
        retrieval_results = {}
        for (i, query) in enumerate(queries):
            query = query.split('/')[-1].split('.')[0]
            sim_list = zip(references, sim_matrix[i].tolist())
            sorted_sim_list = sorted(sim_list, key=lambda x: x[1], reverse=True)
            sorted_sim_list = sorted_sim_list[:1000]
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


if __name__ == '__main__':
    REDUCE_MODE = False     # reduce lr mode
    AUG_MODE = False        # augmentation mode
    PT_MODE = False          # pre trained(imagenet) mode

    EPOCHS = 50
    BATCH_SIZE = 16
    VAL_RATIO = 0.10
    FOLD_NUM = 8
    LR = 0.00005
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
    num_classes = 1383
    num_total = 73551
    input_shape = (224, 224, 3)  # input image shape

    """ Model """
    if PT_MODE:
        model = MobileNetV2(weights='imagenet', input_shape=input_shape, classes=num_classes)
    else:
        model = MobileNetV2(weights='imagenet', input_shape=input_shape, include_top=False)

    last = model.output
    x = GlobalAveragePooling2D()(last)
    base_model = Model(model.input, x)

    model.summary()
    base_model.summary()
    bind_model(base_model)

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True

        """ Initiate Adam optimizer """
        opt = keras.optimizers.Adam(lr=LR)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        """ Load data """
        print('dataset path', DATASET_PATH)
        output_path = ['./img_list.pkl', './label_list.pkl']
        train_dataset_path = DATASET_PATH + '/train/train_data'

        """ Create Generator """
        datagen = ImageDataGenerator(rescale=1./255., validation_split=VAL_RATIO)
        train_gen = datagen.flow_from_directory(train_dataset_path, target_size=input_shape[:2],
                                                batch_size=BATCH_SIZE, subset="training")
        val_gen = datagen.flow_from_directory(train_dataset_path, target_size=input_shape[:2],
                                              batch_size=BATCH_SIZE, subset="validation")

        nsml.save(0)

'''
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
            # Reduce LR
            if REDUCE_MODE:
                if epoch % REDUCE_STEP == REDUCE_STEP - 1:
                    LR *= REDUCE_FACT
                    print("ReduceLR:", LR)
                    opt = keras.optimizers.Adam(lr=LR)
                    model.compile(loss='categorical_crossentropy',
                                  optimizer=opt,
                                  metrics=['accuracy'])


            # Train
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
                res = model.fit_generator(train_gen,
                                          steps_per_epoch=len(train_gen) // 2,
                                          initial_epoch=epoch,
                                          epochs=epoch + 1,
                                          callbacks=[reduce_lr],
                                          validation_data=val_gen,
                                          validation_steps=len(val_gen) // 2,
                                          verbose=1)

            # save & print all logs
            hist_all.append(res.history)
            for i, hist in enumerate(hist_all):
                print(i, hist)
            train_loss, train_acc = res.history['loss'][0], res.history['acc'][0]

            # save model to nsml
            nsml.report(summary=True, epoch=epoch, epoch_total=nb_epoch,
                        loss=train_loss, acc=train_acc)
            while True:
                try:
                    nsml.save(epoch)
                except:
                    print("!!! NSML SAVE ERROR !!!, so retry ")
                    continue
                break
            """
'''


