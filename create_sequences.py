
import os
import pickle
import numpy as np
import pandas as pd

class Sequences:

    def __init__(self, data_type, sequence_split, RANDOM_STATE = 11):

        self.data_type = data_type
        self.sequence_split = sequence_split
        self.RANDOM_STATE = RANDOM_STATE

        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []


def sequence_to_xy_not_combined(seq):
    x = seq[:-1]
    choice = np.nonzero(seq.values)[1]
    y = pd.Series(choice, index=seq.index)
    y = y.astype('int64')[1:]
    return x, y

def sequence_to_xy_combined(seq):
    x = seq.iloc[:-1]
    y = seq['W'].iloc[1:]
    return x, y

def sequence_to_xy_as_vocab(seq):
    return seq[:-1], seq[1:]

#takes a session and returns a list of blocks
def block_split_session(session, mu = 20, sigma = 7, min_seq_length = 7):
    sequences = []
    blocks = session.groupby(axis = 0, level = 'block')
    for label, block in blocks:
        sequences.append(block)
    return sequences

#input: session, returns list of sequences of len ~ N(mu, sigma)
def random_split_session(session,
                         mu = 20,
                         sigma = 7,
                         min_seq_length = 7,
                         overlap = 0):

    k_old, k_new = 0, 0
    finished = 0
    noTrials = len(session)
    sequences = []
    while finished == 0:

        #sample length of sequence from normal distribution
        rand_len = \
                max(min_seq_length, int(np.floor(np.random.normal(mu, sigma))))

        k_new = k_old + rand_len


        sequences.append(session.iloc[k_old:k_new])
        #shifting index
        if overlap == 0:
            k_old = k_new
        elif overlap == 1:
            k_old = k_new - mu / 2

        if noTrials - k_old < mu:
            finished = 1
            #still want to keep min sequence legnth requirements
            #otherwise little chunk of data gets thrown out :'(
            if (noTrials - k_old) > min_seq_length:
                sequences.append(session.iloc[k_old:])
    return sequences


def binary_split_session(session, seq_length = 12):
    noTrials = len(session)



def choice_reward_not_combined(df):
    #including reward information
    choice_reward = df['choice',0] * 2 + df['reward',0]
    choice_reward = choice_reward.astype('int64')
    values = np.zeros([len(choice_reward), len(np.unique(choice_reward))],
                        dtype=int)
    values[np.arange(len(choice_reward)), choice_reward] = 1
    choice_reward = pd.DataFrame(values,
                                 index = df.index,
                                 columns = pd.Index(['E_NR','E_R','W_NR','W_R']))
    return choice_reward


def choice_reward_combined(df):
    #changing var to int type
    choice = df['choice', 0].astype('int64')
    reward = df['reward', 0].astype('int64')
    #converting to 4 binary variables and renaming columns accordingly
    choice_reward = \
        pd.concat([(choice + 1)%2, choice, (reward + 1)%2, reward], axis=1)
    choice_reward.columns = choice_reward.columns.droplevel(level=0)
    choice_reward.columns = pd.Index(['E','W','NR','R'])
    return choice_reward

def choice_reward_as_vocab(df):
    #including reward information
    choice_reward = df['choice',0] * 2 + df['reward',0]
    choice_reward = choice_reward.astype('int64')
    return choice_reward

def add_start_token(seq, start_token):
    #adding start token to each session
    sessions = seq.groupby(axis = 0, level = 'session')
    mod = sessions.get_group('S0')
    mod.loc[pd.IndexSlice['S0', 'B1', -1]] = start_token
    for label, sess in sessions:
        if label != 'S0':
            sess.loc[pd.IndexSlice[label, 'B1', -1]] = start_token
            mod = pd.concat([mod, sess], axis=0)
    mod.sort_index(axis=0, inplace = True)
    return mod
