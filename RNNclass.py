import itertools
import operator
import numpy as np
import sys
import os
import time
from datetime import datetime

class RNN:

    def __init__(self, noFeatures = 4,
                       hidden_dim = 20,
                       bptt_truncate = 4,
                       RANDOM_STATE = 11):
        # Assign instance variables
        self.noFeatures = noFeatures
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.random_state = RANDOM_STATE
        self.loss_sgd = []
        
        # Randomly initialize network parameters between 1/sqrt()
        f = lambda x,y: y * np.sqrt(1./x)

        self.Wxh = np.random.uniform(f(noFeatures, -1), f(noFeatures, 1),
                                                    (hidden_dim, noFeatures))
        self.Why = np.random.uniform(f(hidden_dim, -1), f(hidden_dim, 1),
                                                    (noFeatures, hidden_dim))
        self.Whh = np.random.uniform(f(hidden_dim, -1), f(hidden_dim, 1),
                                                    (hidden_dim, hidden_dim))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


    def predict(self, seq):
        p, h  = self.forward_propagation(seq)
        #max of last prediction
        return np.argmax(p, axis=1)


    def forward_propagation(self, seq):
        noTrials = len(seq)
        #this is the hidden 'space'
        h = np.zeros([1 + noTrials, self.hidden_dim])
        #this is the output - probability of next element in sequence
        p = np.zeros([noTrials, self.noFeatures])

        for trial in range(noTrials):
            h[trial] = np.tanh(self.Wxh.dot(seq[trial]) \
                             + self.Whh.dot(h[trial - 1]))
            p[trial] = self.softmax(self.Why.dot(h[trial]))
        return p, h


    def calculate_total_loss(self, x, y):
        L = 0
        # For each sequence of trials...
        for i in np.arange(len(y)):

            p, h = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_trial_predictions = p[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_trial_predictions))
        return L

    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y) / N


    def bptt(self, x, y, verbose = 0):
        #x and y are NOT one-hot vectors, also they are a single example

        noTrials = len(y)

        # forward propagation
        p, h = self.forward_propagation(x)

        # We accumulate the gradients in these variables
        dLdWxh = np.zeros(self.Wxh.shape)
        dLdWhh = np.zeros(self.Whh.shape)
        dLdWhy = np.zeros(self.Why.shape)

        #difference between prediction and labels
        delta_p = p
        delta_p[np.arange(noTrials), y] -= 1.

        # For each output backwards...
        for trial in np.arange(noTrials)[::-1]:

            #easiest case
            dLdWhy += np.outer(delta_p[trial], h[trial].T)

            # Initial delta calculation
            delta_t = self.Why.T.dot(delta_p[trial]) * (1 - (h[trial] ** 2))
            left_trial = max(0, trial - self.bptt_truncate)
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(left_trial, trial + 1)[::-1]:
                #for debugging
                if verbose: print "Backpropagation step t=%d bptt step=%d " \
                                                        %(trial, bptt_step)
                dLdWhh += np.outer(delta_t, h[bptt_step - 1])

                #generalizes from one hot vectors to inputs with interactions
                for layer in np.nonzero(x[bptt_step])[0]:
                    dLdWxh[:, layer] += delta_t
                # Update delta for next step
                delta_t = self.Whh.T.dot(delta_t) * (1 - h[bptt_step - 1] ** 2)
        return [dLdWxh, dLdWhh, dLdWhy]



    def gradient_check(self, x, y, delta = 0.001, error_threshold = 0.01):

        #i hope this fixes the issues, bro - it did!
        self.bptt_truncate = 1000

        # Calculate the gradients using backpropagation.
        # We want to check if these are correct.
        bptt_gradients = self.bptt(x, y)
        # List of all parameters we want to check.
        model_parameters = ['Wxh', 'Whh', 'Why']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print "Performing gradient check for parameter %s with size %d." \
                                            % (pname, np.prod(parameter.shape))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter,
                           flags=['multi_index'],
                           op_flags=['readwrite'])

            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+delta) - f(x-delta))/(2*delta)
                parameter[ix] = original_value + delta
                gradplus = self.calculate_total_loss([x], [y])
                parameter[ix] = original_value - delta
                gradminus = self.calculate_total_loss([x], [y])
                estimated_gradient = (gradplus - gradminus) / (2 * delta)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient) \
                    / (np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is too large -> fail the gradient check
                if relative_error > error_threshold:
                    print "Gradient Check ERROR: parameter=%s ix=%s" %(pname,ix)
                    print "+h Loss: %f" % gradplus
                    print "-h Loss: %f" % gradminus
                    print "Estimated_gradient: %f" % estimated_gradient
                    print "Backpropagation gradient: %f" % backprop_gradient
                    print "Relative Error: %f" % relative_error
                    return
                it.iternext()
            print "Gradient check for parameter %s passed." % (pname)





    def numpy_sdg_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdWxh, dLdWhh, dLdWhy = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.Wxh -= learning_rate * dLdWxh
        self.Why -= learning_rate * dLdWhy
        self.Whh -= learning_rate * dLdWhh

    def train_with_sgd(model, X_train, y_train, learning_rate=0.005,
                       nepoch=100, evaluate_loss_after=5, verbose=0):
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            # Optionally evaluate the loss
            if (epoch % evaluate_loss_after == 0):
                loss = model.calculate_loss(X_train, y_train)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if verbose == 1:
                    print "%s: Loss after num_examples_seen=%d epoch=%d: %f" \
                                        %(time, num_examples_seen, epoch, loss)
                # Adjust the learning rate if loss increases
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5
                    if verbose == 1:
                        print "Setting learning rate to %f" % learning_rate
                sys.stdout.flush()
            # For each training example...
            for i in range(len(y_train)):
                # One SGD step
                model.numpy_sdg_step(X_train[i], y_train[i], learning_rate)
                num_examples_seen += 1

        self.loss_sgd = losses
