import numpy as np
import matplotlib.pyplot as plt


def input_length_distribution(seqs):
    lengths = [len(w) for w in seqs]
    bins = np.linspace(0, max(lengths), 10)
    plt.hist(lengths, bins=bins)
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.title('Input Sequence Length Distribution')
    plt.show()


def scatter_sequence_length_to_accuracy(slta, split_type):
    #slta is list of pairs (seq_length, performance)
    seq_lengths = [w[0] for w in seq_length_to_accuracy]
    seq_score = [w[1] for w in seq_length_to_accuracy]
    plt.scatter(seq_lengths, seq_score, marker = 'o', color = 'green')
    plt.title('RNN Correlation - %s' %split_type)
    plt.xlabel('Sequence Length')
    plt.ylabel('Decoding Accuracy')
    plt.show()
