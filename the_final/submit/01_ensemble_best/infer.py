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


def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(file_path):
        model.load_weights(file_path)
        print('model loaded!')

    def infer(queries, _):
        print('infer')
        test_path = DATASET_PATH + '/test/test_data'

        db = [os.path.join(test_path, 'reference', path) for path in os.listdir(os.path.join(test_path, 'reference'))]

        queries = [v.split('/')[-1].split('.')[0] for v in queries]
        db = [v.split('/')[-1].split('.')[0] for v in db]
        queries.sort()
        db.sort()

        queries, query_vecs, references, reference_vecs = get_feature(model, queries, db)

        # l2 normalization
        # query_vecs = l2_normalize(query_vecs)
        # reference_vecs = l2_normalize(reference_vecs)

        # Calculate cosine similarity
        sim_matrix = np.dot(query_vecs, reference_vecs.T)
        indices = np.argsort(sim_matrix, axis=1)
        indices = np.flip(indices, axis=1)

        # query expansion
        expansion_step = 10
        m_sample = 5
        for _ in range(expansion_step):
            for i, ind in enumerate(indices):
                query_vecs[i] = np.mean(
                    np.vstack(
                        (query_vecs[i], reference_vecs[ind[:m_sample]])
                    ), axis=0)
            # recalculate (cos-sim)
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
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, norm, where=norm!=0)
