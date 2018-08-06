import numpy as np

def save_model_parameters(outfile, model):
    Wxh, Why, Whh = model.Wxh, model.Why, model.Whh
    np.savez(outfile, Wxh = Wxh, Why = Why, Whh = Whh)
    print "Saved model parameters to %s." %outfile
    return


def load_model_parameters(path, model):
    npzfile = np.load(path)
    Wxh, Why, Whh = npzfile["Wxh"], npzfile["Why"], npzfile["Whh"]
    model.hidden_dim = Wxh.shape[0]
    model.noFeatures = Wxh.shape[1]
    model.Wxh = Wxh
    model.Why = Why
    model.Whh = Whh
    print "Loaded model parameters from %s. hidden_dim=%d word_dim=%d" \
                                    % (path, Wxh.shape[0], Wxh.shape[1])
    return
