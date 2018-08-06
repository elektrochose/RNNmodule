
import os
import copy
import pickle
import numpy as np
import pandas as pd
idx = pd.IndexSlice


class Sequences:

    def __init__(self, data_type, sequence_split, RANDOM_STATE = 11):

        self.data_type = data_type
        self.sequence_split = sequence_split
        self.RANDOM_STATE = RANDOM_STATE
        self.position = 0

        self.X_train = []
        self.y_train = []
        self.X_validate = []
        self.y_validate = []
        self.X_test = []
        self.y_test = []

    def train_validate_test_split_by_session(self, df,
                                             validate_size = 0.15,
                                             test_size = 0.15):

        sessions = df.groupby(axis = 0, level = 'session')
        noSessions = len(sessions)
        shuffleIndex = np.arange(noSessions)
        np.random.seed(self.RANDOM_STATE)
        np.random.shuffle(shuffleIndex)
        cutoff1 = int(np.floor(noSessions * validate_size))
        cutoff2 = int(np.floor(noSessions * test_size))
        validate = ['S%i' %s for s in shuffleIndex[:cutoff1]]
        test = ['S%i' %s for s in shuffleIndex[cutoff1: cutoff1 + cutoff2]]
        train = ['S%i' %s for s in shuffleIndex[cutoff1 + cutoff2:]]
        return df.loc[idx[train, :, :], :], \
               df.loc[idx[validate, :, :], :], \
               df.loc[idx[test, :, :], :]

    def reduce_features_omni(self, df):
        out = []
        df['block_counter'] = df.index.labels[2]
        out.append(df.loc[:, idx['block_counter']])
        df['trial_counter'] = np.arange(len(df))
        out.append(df.loc[:,idx['trial_counter']])
        out.append(df['GA', 0])
        out.append(df['SA', 0])
        #current and next choice
        out.append(df.loc[:,idx['choice', [0]]])

        future = df.loc[:,idx['choice', [0]]].shift(-1)
        future.columns.set_levels([1],level='trials_ago',inplace = True)
        out.append(future)
        #reward information
        out.append(df.loc[:, idx['reward',0]])
        for time_label in ['RT1', 'RT2']:
            #centering timing information
            RT1 = (df.loc[:, idx[time_label,0]] \
                - np.mean(df.loc[:, idx[time_label,0]])) \
                / np.std(df.loc[:, idx[time_label,0]])
            #breaking it up into quartiles, since std == 1, we can exclude outliers
            #by only discretizing within 2 standard deviations: np.linspace...
            RT1 = pd.Series(np.digitize(RT1,
                                        np.linspace(-2, 2, 10)),
                                        index = RT1.index, name = time_label)
            out.append(RT1)

        out = pd.concat(out, axis=1)
        out = out.rename(columns={('choice', 0):'choice',
                                  ('choice', 1): 'y',
                                  ('reward', 0) : 'reward',
                                  ('GA', 0) : 'GA',
                                  ('SA', 0) : 'SA'})
        out = out.dropna().astype('int16')
        return out

    def reduce_features(self, df):
        out = []
        #block information
        df['trial_counter'] = np.arange(len(df))
        out.append(df.loc[:,idx['trial_counter']])
        #current and next choice
        out.append(df.loc[:,idx['choice', [0]]])

        future = df.loc[:,idx['choice', [0]]].shift(-1)
        future.columns.set_levels([1],level='trials_ago',inplace = True)
        out.append(future)
        #reward information
        out.append(df.loc[:, idx['reward',0]])
        for time_label in ['RT1', 'RT2']:
            #centering timing information
            RT1 = (df.loc[:, idx[time_label,0]] \
                - np.mean(df.loc[:, idx[time_label,0]])) \
                / np.std(df.loc[:, idx[time_label,0]])
            #breaking it up into quartiles, since std == 1, we can exclude outliers
            #by only discretizing within 2 standard deviations: np.linspace...
            RT1 = pd.Series(np.digitize(RT1,
                                        np.linspace(-2, 2, 10)),
                                        index = RT1.index, name = time_label)
            out.append(RT1)
        out = pd.concat(out, axis=1)
        out = out.rename(columns={('choice',0):'choice',
                                  ('choice',1):'y',
                                  ('reward',0) : 'reward'})
        return out.dropna().astype('int16')

    def reduce_features_minimal(self, df):
        out = []
        #current and next choice
        out.append(df.loc[:,idx['choice', [0]]])

        future = df.loc[:,idx['choice', [0]]].shift(-1)
        future.columns.set_levels([1],level='trials_ago',inplace = True)
        out.append(future)
        #reward information
        out.append(df.loc[:, idx['reward',0]])
        out = pd.concat(out, axis=1)

        tmp_out = []
        tmp_out.append(out['choice'].rename(columns={0:'choice',1:'y'}))
        tmp_out.append(out['reward'].rename(columns={0:'reward'}))
        out = pd.concat(tmp_out, axis = 1)

        return out.dropna().astype('int16')

    def reduce_features_minimal_onehot(self, df):
        combined =  df.loc[:,idx['choice', 0]] * 2 + df.loc[:, idx['reward',0]]
        combined = combined.to_frame(name = 'x')

        combined['y'] = combined.shift(-1)
        combined = combined.dropna().astype('int16')
        all_vals = np.reshape([v for v in range(4) for w in range(2)], (4,2))
        combined_ext = pd.concat([combined, pd.DataFrame(all_vals,
                                        columns = combined.columns)], axis =0)
        X = pd.get_dummies(combined_ext['x'])
        y = pd.get_dummies(combined_ext['y'])
        cols = pd.MultiIndex.from_product([['x','y'],range(4)])

        out = pd.DataFrame(np.concatenate((X.values[:-4], y.values[:-4]), axis=1),
                           index = combined.index,
                           columns = cols)
        return out

    def split_session_into_sequences(self, session, timesteps = 20):

        samples = len(session)
        features = session.shape[1] - 1

        raw_data = session.drop(axis = 1, labels = 'y').values
        X = np.full([samples, timesteps, features], np.NaN)
        y = session['y']

        for sample in range(samples):

            if sample < timesteps: #padding needed
                tmp = np.zeros([timesteps, features])
                chunk = raw_data[:sample + 1, :] #inclusive of current trial
                tmp[- 1 + sample * -1:,:] = chunk
                X[sample, :, :] = tmp
            elif sample >= timesteps:
                X[sample, :, :] =  \
                        raw_data[sample - timesteps + 1: sample + 1, :]
        return X, y

    def split_session_into_sequences_oneHot(self, session, timesteps = 20):

        samples = len(session)
        features = session.shape[1] - 4

        raw_data = session.drop(axis = 1, labels = 'y').values
        X = np.full([samples, timesteps, features], np.NaN)
        y = session['y']

        for sample in range(samples):

            if sample < timesteps: #padding needed
                tmp = np.zeros([timesteps, features])
                chunk = raw_data[:sample + 1, :] #inclusive of current trial
                tmp[- 1 + sample * -1:,:] = chunk
                X[sample, :, :] = tmp
            elif sample >= timesteps:
                X[sample, :, :] =  \
                        raw_data[sample - timesteps + 1: sample + 1, :]
        return X, y


    def create_sequences(self, df, timesteps = 20,
                                   feature_dim = 'binaryMinimal',
                                   validate_size = 0.25,
                                   test_size = 0.25):

        def aggregate_by_session(split):
            X = []
            y = []
            sessions = split.groupby(axis = 0, level = 'session')
            for label, session in sessions:
                if not feature_dim.find('binaryMinimal') < 0:
                    RF_func = self.reduce_features_minimal
                elif not feature_dim.find('binaryMid') < 0:
                    RF_func = self.reduce_features
                elif not feature_dim.find('binaryOmni') < 0:
                    RF_func = self.reduce_features_omni
                elif not feature_dim.find('OneHotBinaryMinimal') < 0:
                    RF_func = self.reduce_features_minimal_onehot

                session = RF_func(session)

                if not feature_dim.find('OneHot') < 0:
                    tmp_X = session['x']
                    tmp_y = session['y']
                    tmp_X, tmp_y = \
                        self.split_session_into_sequences_oneHot(session,
                                                          timesteps = timesteps)
                else:
                    tmp_X, tmp_y = \
                        self.split_session_into_sequences(session,
                                                          timesteps = timesteps)
                X.append(tmp_X)
                y.append(tmp_y)
            X = np.concatenate(X)
            y = np.concatenate(y)
            return X,y

        train, validate, test = \
            self.train_validate_test_split_by_session(df,
                                            validate_size = validate_size,
                                            test_size = test_size)

        mirror = copy.deepcopy(train)
        mirror['choice',0] = (mirror['choice',0] + 1) % 2
        session_labels = [w + '_mirror' for w in mirror.index.levels[0]]
        mirror.index.set_levels(session_labels, level='session', inplace = True)
        train = pd.concat([train, mirror], axis=0)

        X_train, y_train = aggregate_by_session(train)
        self.X_train = X_train
        self.y_train = y_train

        X_validate, y_validate = aggregate_by_session(validate)
        self.X_validate = X_validate
        self.y_validate = y_validate

        X_test, y_test = aggregate_by_session(test)
        self.X_test = X_test
        self.y_test = y_test

        return

    def feed_next(self, batch_size = 32):
        if self.position <= self.X_train.shape[0] - batch_size:
            out = self.X_train[self.position: self.position + batch_size, :, :]
            self.position += batch_size
            return out
        else:
            return None
