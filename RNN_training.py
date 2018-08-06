#THIS IS EPSILON ROOT DIRECTORY !
ROOT = '/home/pablo/python/'

#standard modules
import sys
if ROOT not in sys.path:
    sys.path.append(ROOT)

import pandas as pd
import numpy as np
import os
import pickle
import time
import itertools
import multiprocessing as mp
#my modules
from behavioral_performance.utils import *
from RNNmodule.RNNclass import RNN
from RNNmodule.create_sequences import Sequences
print 'done loading modules...'


RANDOM_STATE = 11
data_dir = ROOT + 'DATA_structures/TbyT/'
sequence_dir = ROOT + 'DATA_structures/RNN_sequences'
model_dir = ROOT + 'Models/RNN'
data_type = ['combined_input', 'not_combined']
sequence_split = ['overlapping', 'non_overlapping']
sequence_length = ['long','short']
hidden_dimensions = [2,4,5,8,10,15,20,50,100,200]



def train_network(sequence_path, model_path, hidden_dim = 20, nepoch = 100):
    if not os.path.isfile(sequence_path):
        print 'WARNING: could not find sequences'
    if not os.path.isdir(model_path[:-model_path[::-1].find('/')]):
        print 'WARNING: model path file is invalid'

    print 'training network: %s' %sequence_path
    seqObject = pickle.load(open(sequence_path, 'rb'))

    #instantiate RNN for that set of sequences
    myRNN = RNN(noFeatures = 4,
                hidden_dim = hidden_dim,
                bptt_truncate = 30,
                RANDOM_STATE = seqObject.RANDOM_STATE)
    #train the network
    myRNN.train_with_sgd([seq.values for seq in seqObject.X_train],
                         [seq.values for seq in seqObject.y_train],
                         learning_rate = 0.005,
                         nepoch = nepoch,
                         verbose = 0)
    #save the network
    pickle.dump(myRNN, open(model_path, 'wb'))



pool = mp.Pool(processes=32)

for fileName, dt, ss, sl, hd in \
    itertools.product(fileNames, data_type, sequence_split, sequence_length, hidden_dimensions):
    #load appropriate sequences
    sequence_path = '/'.join([sequence_dir, dt, 'random_lengths', ss, sl, fileName])
    model_path = '/'.join([model_dir, dt, 'random_lengths', ss, sl,
                        fileName[:fileName.find('.')] + str(hd) + '.p'])
    pool.apply_async(train_network, [sequence_path, model_path, hd, 300])
