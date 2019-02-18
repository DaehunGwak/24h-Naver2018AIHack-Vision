# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time

import nsml
import numpy as np

from nsml import DATASET_PATH
import keras
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics.pairwise import euclidean_distances
from model import *
from infer import *


def get_train_datagen(aug_mode=False, val_mode=False, val_ratio=0.1):
    datagen = None
    if aug_mode:
        if val_mode:
            datagen = ImageDataGenerator(
                rescale=1. / 255,
                validation_split=val_ratio,
                horizontal_flip=True,
                shear_range=0.05,
                zoom_range=0.1,
                rotation_range=10
            )
        else:
            datagen = ImageDataGenerator(
                rescale=1. / 255,
                horizontal_flip=True,
                shear_range=0.05,
                zoom_range=0.1,
                rotation_range=10
            )
    else:
        if val_mode:
            datagen = ImageDataGenerator(
                rescale=1. / 255,
                validation_split=val_ratio
            )
        else:
            datagen = ImageDataGenerator(
                rescale=1. / 255
            )
    return datagen


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--nb_epoch', type=int, default=1000)
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--num_classes', type=int, default=1383)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    config = args.parse_args()

    # training parameters
    aug_mode = False
    val_mode = False
    weight_mode = 'imagenet'
    val_ratio = 0.1
    learning_rate = 0.0001
    pre_epoch = 5
    nb_epoch = config.nb_epoch
    batch_size = 8
    num_classes = config.num_classes
    input_shape = (224, 224, 3)  # input image shape

    """ Model """
    base_model, embedding_model, model = get_siamese_model(input_shape=input_shape, embedding_dim=2048, weight_mode=weight_mode)
    embedding_model.summary()
    model.summary()
    set_embedding_model(embedding_model)    # for generator
    bind_model(embedding_model)             # for nsml

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True

        """ Initiate Adam optimizer """
        opt = keras.optimizers.Adam(lr=learning_rate)
        model.compile(loss=None,
                      optimizer=opt,
                      metrics=['accuracy'])

        print('dataset path', DATASET_PATH)

        train_datagen = get_train_datagen(aug_mode=aug_mode,
                                          val_mode=val_mode,
                                          val_ratio=val_ratio)

        train_gen = train_datagen.flow_from_directory(
            directory=DATASET_PATH + '/train/train_data',
            target_size=input_shape[:2],
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42
        )
        TRAIN_PATH = DATASET_PATH + '/train/train_data/'
        train_generator = generate_samples(train_datagen, train_gen, batch_size, TRAIN_PATH,
                                           input_shape=input_shape,
                                           train_mode="training")
        if val_mode:
            val_generator = generate_samples(train_datagen, train_gen, batch_size, TRAIN_PATH,
                                             input_shape=input_shape,
                                             train_mode="validation",
                                             random_mode=True)

        """ Callback """
        # monitor = 'acc'
        # reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)

        """ Training loop """
        # train classification
        pre_model = add_classification_dense_model(base_model, num_classes=num_classes)
        pre_model.compile(loss='categorical_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])
        pre_hist_all = []
        for epoch in range(pre_epoch):
            res = pre_model.fit_generator(generator=train_gen,
                                          steps_per_epoch=len(train_gen),
                                          initial_epoch=epoch,
                                          epochs=epoch + 1,
                                          verbose=1,
                                          shuffle=True)
            pre_hist_all.append(res.history)
            for i, hist in enumerate(pre_hist_all):
                print(i, hist)

        # freezing
        for layer in embedding_model.layers[:-2]:
            layer.trainable = False
        for i, layer in enumerate(embedding_model.layers):
            print(i, layer.name, layer.trainable)
        model.compile(loss=None,
                      optimizer=opt,
                      metrics=['accuracy'])

        # train by siamese triplet loss
        STEP_SIZE_TRAIN = num_classes // batch_size
        t0 = time.time()
        hist_all = []
        for epoch in range(nb_epoch):
            t1 = time.time()
            if val_mode:
                res = model.fit_generator(generator=train_generator,
                                          steps_per_epoch=STEP_SIZE_TRAIN,
                                          validation_data=val_generator,
                                          validation_steps=5,
                                          initial_epoch=epoch,
                                          epochs=epoch + 1,
                                          verbose=1,
                                          shuffle=True,
                                          max_queue_size=1,
                                          workers=0,
                                          use_multiprocessing=False)
            else:
                res = model.fit_generator(generator=train_generator,
                                          steps_per_epoch=STEP_SIZE_TRAIN,
                                          initial_epoch=epoch,
                                          epochs=epoch + 1,
                                          verbose=1,
                                          shuffle=True,
                                          max_queue_size=1,
                                          workers=0,
                                          use_multiprocessing=False)
            t2 = time.time()
            hist_all.append(res.history)
            for i, hist in enumerate(hist_all):
                print(i, hist)
            print('Training time for one epoch : %.1f' % (t2 - t1))
            train_loss = res.history['loss'][0]
            nsml.report(summary=True, epoch=epoch, epoch_total=nb_epoch, loss=train_loss)
            while True:
                try:
                    nsml.save(epoch)
                except:
                    print("!!! NSML SAVE ERROR !!!, so retry ")
                    continue
                break
        print('Total training time : %.1f' % (time.time() - t0))

