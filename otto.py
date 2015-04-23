import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from nolearn.lasagne import NeuralNet
from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax

# Utility functions
def load_train_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.values.copy()
    # Shuffle the training data
    np.random.shuffle(X)
    # Extract labels, toss IDs
    X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, scaler, encoder

def load_test_data(csv_path, scaler):
    df = pd.read_csv(csv_path)
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    X = scaler.transform(X)
    return X, ids

def make_submission(classifier, X_test, ids, encoder, output_file='mysubmission.csv.gz'):
    y_prob = classifier.predict_proba(X_test)
    import gzip
    with gzip.open(output_file, 'w') as output:
        output.write('id,')
        output.write(','.join(encoder.classes_))
        output.write('\n')
        for id, probs in zip(ids, y_prob):
            line = ','.join([id] + map(str, probs.tolist()))
            output.write(line)
            output.write('\n')
    print('Wrote submission to file {}'.format(output_file))

# Read in our data
X, y, scaler, encoder = load_train_data('data/train.csv')
X_test, ids = load_test_data('data/test.csv', scaler)
num_classes = len(encoder.classes_)
num_features = X.shape[1]

# Build our neural network
netlayers = [('input', layers.InputLayer),
             ('hidden0', layers.DenseLayer),
             ('dropout', layers.DropoutLayer),
             ('hidden1', layers.DenseLayer),
             ('output', layers.DenseLayer)]

net = NeuralNet(layers=netlayers,
                input_shape=(None, num_features),

                hidden0_num_units=800,
                dropout_p=0.7,
                hidden1_num_units=800,
                output_num_units=num_classes,
                output_nonlinearity=softmax,

                update=nesterov_momentum,
                update_learning_rate=0.01,
                update_momentum=0.8,

                eval_size=0.2,
                max_epochs=40,
                verbose=1)

# Fit the data
net.fit(X, y)

# Output the results
make_submission(net, X_test, ids, encoder)
