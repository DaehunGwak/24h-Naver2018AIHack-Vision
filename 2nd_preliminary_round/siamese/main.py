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
from model import get_model

def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(file_path):
        model.load_weights(file_path)
        print('model loaded!')

    def infer(queries, _):
        test_path = DATASET_PATH + '/test/test_data'

        db = [os.path.join(test_path, 'reference', path) for path in os.listdir(os.path.join(test_path, 'reference'))]

        queries = [v.split('/')[-1].split('.')[0] for v in queries]
        db = [v.split('/')[-1].split('.')[0] for v in db]
        queries.sort()
        db.sort()

        queries, query_vecs, references, reference_vecs = get_feature(model, queries, db)

        # l2 normalization
        query_vecs = l2_normalize(query_vecs)
        reference_vecs = l2_normalize(reference_vecs)

        # Calculate cosine similarity
        sim_matrix = np.dot(query_vecs, reference_vecs.T)
        indices = np.argsort(sim_matrix, axis=1)
        indices = np.flip(indices, axis=1)

        retrieval_results = {}

        for (i, query) in enumerate(queries):
            ranked_list = [references[k] for k in indices[i]]
            ranked_list = ranked_list[:1000]

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
def get_feature(model, queries, db):
    img_size = (224, 224)
    test_path = DATASET_PATH + '/test/test_data'

    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('output_layer').output)
    test_datagen = ImageDataGenerator(rescale=1. / 255, dtype='float32')
    query_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=(224, 224),
        classes=['query'],
        color_mode="rgb",
        batch_size=32,
        class_mode=None,
        shuffle=False
    )
    query_vecs = intermediate_layer_model.predict_generator(query_generator, steps=len(query_generator), verbose=1)

    reference_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=(224, 224),
        classes=['reference'],
        color_mode="rgb",
        batch_size=32,
        class_mode=None,
        shuffle=False
    )
    reference_vecs = intermediate_layer_model.predict_generator(reference_generator, steps=len(reference_generator),
                                                                verbose=1)

    return queries, query_vecs, db, reference_vecs


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epoch', type=int, default=5)
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--num_classes', type=int, default=1383)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    config = args.parse_args()

    # training parameters
    val_ratio = 0.1
    learning_rate = 0.00045
    nb_epoch = 20
    batch_size = config.batch_size
    num_classes = config.num_classes
    input_shape = (224, 224, 3)  # input image shape

    """ Model """
    model = get_model(input_shape=input_shape, num_classes=num_classes, weight_mode=None)
    model.summary()
    bind_model(model)

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True

        """ Initiate Adam optimizer """
        opt = keras.optimizers.Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        print('dataset path', DATASET_PATH)

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=val_ratio,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        train_generator = train_datagen.flow_from_directory(
            directory=DATASET_PATH + '/train/train_data',
            subset='training',
            target_size=input_shape[:2],
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42
        )
        val_generator = train_datagen.flow_from_directory(
            directory=DATASET_PATH + '/train/train_data',
            subset='validation',
            target_size=input_shape[:2],
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42
        )
        print(train_generator.class_indices)
        class_list = list(train_generator.class_indices.keys())
        TRAIN_PATH = DATASET_PATH + '/train/train_data/'
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
        """ Callback """
        monitor = 'acc'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)

        """ Training loop """
        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        t0 = time.time()
        hist_all = []
        for epoch in range(nb_epoch):
            t1 = time.time()
            res = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=len(train_generator),
                                      validation_data=val_generator,
                                      validation_steps=len(val_generator),
                                      initial_epoch=epoch,
                                      epochs=epoch + 1,
                                      callbacks=[reduce_lr],
                                      verbose=1,
                                      shuffle=True)
            t2 = time.time()
            hist_all.append(res.history)
            for i, hist in enumerate(hist_all):
                print(i, hist)
            print('Training time for one epoch : %.1f' % (t2 - t1))
            train_loss, train_acc = res.history['loss'][0], res.history['acc'][0]
            nsml.report(summary=True, epoch=epoch, epoch_total=nb_epoch, loss=train_loss, acc=train_acc)
            while True:
                try:
                    nsml.save(epoch)
                except:
                    print("!!! NSML SAVE ERROR !!!, so retry ")
                    continue
                break
        print('Total training time : %.1f' % (time.time() - t0))
        '''
