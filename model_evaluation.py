import matplotlib.pyplot as plt
import numpy as np

def sequence_to_labels(sequences):
    task = sequences[0] + 'SR'
    regime = sequences[3: sequences.find('_')]
    if sequences.find('reward') > 0:
        seq_type = 'choice_reward'
    elif sequences.find('reward') < 0:
        seq_type = 'choice_only'
    if sequences.find('split') > 0:
        seq_split = 'split'
    elif sequences.find('split') < 0:
        seq_split = 'whole'
    return task, regime, seq_type, seq_split


def evaluate_model(test_seq, seq_type, model):

    noTrials = len(test_seq)
    prediction = np.zeros(noTrials)

    for trial in range(1, noTrials):
        p = model.predict(test_seq[:trial])
        prediction[trial] = p[-1]

    if seq_type == 'choice_only':
        #first element of sequence was given so must subtract 1
        hits = np.sum(test_seq == prediction) - 1
        return hits, (noTrials - 1)
    elif seq_type == 'choice_reward':
        #get a sense of how well sequence did
        sequence_hits = np.sum(test_seq == prediction) - 1
        #we care if choice got decoded, not choice AND reward
        task_hits = 0
        for trial in range(1, noTrials):
            if test_seq[trial] < 2 and prediction[trial] < 2:
                task_hits += 1
            elif test_seq[trial] > 1 and prediction[trial] > 1:
                task_hits += 1
        return task_hits, (noTrials - 1), sequence_hits






#returns hits and noTrials for a given pair of x and y test sequences
#inputs must be values! not dataframes
def test_model_on_test_set_combined(model, X_test, y_test):
    noTrials = len(X_test)
    prediction = np.zeros(noTrials)

    for trial in range(1, noTrials):
        p = model.predict(X_test[:trial])
        prediction[trial] = p[-1]

    hits = np.sum(prediction[1:] == y_test[:-1])
    return hits, noTrials - 1, prediction

def test_model_on_test_not_combined(model, X_test, y_test):
    noTrials = len(X_test)
    prediction = np.zeros(noTrials)

    for trial in range(1, noTrials):
        p = model.predict(X_test[:trial])
        prediction[trial] = p[-1]

    #first, we must collapse to represent only choices
    prediction[prediction < 2] = 0
    prediction[prediction > 1] = 1
    y_test[y_test < 2] = 0
    y_test[y_test > 1] = 1

    hits = np.sum(prediction[1:] == y_test[:-1])
    return hits, noTrials - 1, prediction
