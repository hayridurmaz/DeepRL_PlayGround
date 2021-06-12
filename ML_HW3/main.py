import os
import pickle
from contextlib import redirect_stdout
from time import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import comb
from sklearn import metrics
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import GridSearchCV

import argparse


def getRandDataset(N=1000, R=0.7, class_val=0):
    """
    Returns random array of features and labels
    """
    C = np.random.rand(N, N)
    y = np.full(N, class_val)
    N_times_R = int(N * R)
    return C[:N_times_R], y[:N_times_R], C[N_times_R - N:], y[N_times_R - N:]


def getNotRandDataset(N=1000, R=0.7):
    """
    Returns random array of features and labels
    """
    X, y_true = make_blobs(n_samples=N, centers=2,
                           cluster_std=0.60, random_state=0)
    N_times_R = int(N * R)
    return X[:N_times_R], y_true[:N_times_R], X[N_times_R - N:], y_true[N_times_R - N:]


#
# Perceptron implementation
#
class CustomPerceptron(object):

    def __init__(self, n_iterations=100, random_state=1, learning_rate=0.01):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.learning_rate = learning_rate

    '''
    Stochastic Gradient Descent

    1. Weights are updated based on each training examples.
    2. Learning of weights can continue for multiple iterations
    3. Learning rate needs to be defined
    '''

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.coef_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        for _ in range(self.n_iterations):
            for xi, expected_value in zip(X, y):
                predicted_value = self.predict(xi)
                self.coef_[1:] = self.coef_[1:] + self.learning_rate * (expected_value - predicted_value) * xi
                self.coef_[0] = self.coef_[0] + self.learning_rate * (expected_value - predicted_value) * 1

    '''
    Net Input is sum of weighted input signals
    '''

    def net_input(self, X):
        weighted_sum = np.dot(X, self.coef_[1:]) + self.coef_[0]
        return weighted_sum

    '''
    Activation function is fed the net input and the unit step function
    is executed to determine the output.
    '''

    def activation_function(self, X):
        weighted_sum = self.net_input(X)
        return np.where(weighted_sum >= 0.0, 1, 0)

    '''
    Prediction is made on the basis of output of activation function
    '''

    def predict(self, X):
        return self.activation_function(X)

    '''
    Model score is calculated based on comparison of
    expected value and predicted value
    '''

    def score(self, X, y):
        labels_predicted = []
        misclassified_data_count = 0
        for xi, target in zip(X, y):
            output = self.predict(xi)
            labels_predicted.append(output)
            if (target != output):
                misclassified_data_count += 1
        total_data_count = len(X)
        self.score_ = (total_data_count - misclassified_data_count) / total_data_count
        return self.score_, labels_predicted


"""
Neural net functions started
"""


def testNeuralNets(N, train_data, train_labels, test_data, test_labels):
    # Learning rate
    alpha = 0.01

    prcptrn = CustomPerceptron()
    #
    # Fit the model
    #
    prcptrn.fit(train_data, train_labels)
    #
    # Score the model
    #
    score_val, y_predicted = prcptrn.score(test_data, test_labels)
    plotData(test_data, y_predicted, title="NN Results")
    print("---------NEURAL NETS---------")
    print("SCORE:", score_val)
    print("---------NEURAL NETS---------")


"""
SVM functions started
"""


def getSVMBestEstimator(train_data, train_labels):
    t0 = time()
    # Create a dictionary of possible parameters
    params_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                   'gamma': [0.0001, 0.001, 0.01, 0.1],
                   'kernel': ['linear', 'rbf']}

    # Create the GridSearchCV object
    grid_clf = GridSearchCV(svm.SVC(class_weight='balanced'), params_grid)

    # Fit the data with the best possible parameters
    grid_clf = grid_clf.fit(train_data, train_labels)

    # Print the best estimator with it's parameters
    print(grid_clf.best_estimator_)
    print("Best estimator done in %0.3fs" % (time() - t0))
    return grid_clf


def testSVM(train_data, train_labels, test_data, test_labels):
    if args.best_estimator:
        # Get svm with best estimators (Comment out if not necessary)
        clf = getSVMBestEstimator(train_data, train_labels)
    else:
        # Create a svm Classifier
        clf = svm.SVC(C=0.1, class_weight='balanced', gamma=0.0001, kernel='linear')  # Linear Kernel
        # Train SVM using the training sets
        clf.fit(train_data, train_labels)

    # Predict the response for test dataset
    y_pred = clf.predict(test_data)
    # Model Accuracy calculated
    plotData(test_data, y_pred, title="SVM Result")
    print("---------SVM---------")
    print("SVM Accuracy:", metrics.accuracy_score(test_labels, y_pred))
    precision, recall, fscore, support = score(test_labels, y_pred)
    # print('precision: {}'.format(precision))
    # print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    # print('support: {}'.format(support))
    print("---------SVM---------")

    """
    K-Means functions started
    """


def purity_score(y_true, y_pred):
    """
    K-Means purity calculator
    """
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def rand_index_score(clusters, classes):
    """
    K-Means rand index score calculator
    """
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)


def testKMeans(train_data, train_labels, test_data, test_labels):
    """
    K-Means test method
    """
    km = KMeans(n_clusters=2, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    km.fit(train_data)
    y_pred = km.predict(test_data)
    plotData(test_data, y_pred, title="KMeans Result")
    print("---------KMEANS---------")
    # print("K-Means Accuracy:", metrics.accuracy_score(test_labels, y_pred))
    print("K-Means Purity: ", purity_score(y_true=test_labels, y_pred=y_pred))
    print("K-Means Rand-index score: ", rand_index_score(test_labels, y_pred))
    print("---------KMEANS---------")


def plotData(x, y, title=None, x_label=None, y_label=None):
    # Getting unique labels
    u_labels = np.unique(y)
    # plotting the results:
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for i in u_labels:
        plt.scatter(x[y == i, 0], x[y == i, 1], label=i)
    plt.legend()
    plt.show()


def main():
    dataset_file_name = "data.pkl"
    if args.no_random:
        dataset_file_name = "data_not_random.pkl"

    # Initial config params set
    N = 1000
    R = 0.7
    data = {}
    if not os.path.exists(dataset_file_name):
        if args.no_random:
            C, Y, C_test, Y_test = getRandDataset(N, R)
            # Concat C1 and C2 and their labels
            data = {
                "both_train_data": C,
                "both_train_labels": Y,
                "both_test_data": C_test,
                "both_test_labels": Y_test
            }
        else:
            # If dataset not created, create with random
            # Dataset generation
            C1, Y1, C1_test, Y1_test = getRandDataset(N, R, 1)
            C2, Y2, C2_test, Y2_test = getRandDataset(N, R, 2)
            # Concat C1 and C2 and their labels
            data = {
                "both_train_data": np.concatenate((C1, C2), axis=0),
                "both_train_labels": np.concatenate((Y1, Y2), axis=0),
                "both_test_data": np.concatenate((C1_test, C2_test), axis=0),
                "both_test_labels": np.concatenate((Y1_test, Y2_test), axis=0)
            }
        output = open(dataset_file_name, 'wb')
        pickle.dump(data, output)
        output.close()
    else:
        # If dataset already exists, read it back
        pkl_file = open(dataset_file_name, 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()

    # Visualize data
    plotData(data["both_train_data"], data["both_train_labels"], title="DATASET")

    # SVM TEST
    testSVM(data["both_train_data"], data["both_train_labels"], data["both_test_data"], data["both_test_labels"])

    # K-MEANS TEST
    testKMeans(data["both_train_data"], data["both_train_labels"], data["both_test_data"], data["both_test_labels"])

    # NEURAL NETS TEST
    testNeuralNets(N, data["both_train_data"], data["both_train_labels"], data["both_test_data"],
                   data["both_test_labels"])


parser = argparse.ArgumentParser(description='Test some ML algorithms')
parser.add_argument("-n", "--no-random", action="store_true",
                    help="Create dataset with make_blobs instead of random")
parser.add_argument("-b", "--best-estimator", action="store_true",
                    help="While testing svm, using best estimator instead of pre-defined ones.(Takes more time)")
parser.add_argument("-f", "--file-output", action="store_true",
                    help="Write output to a file instead of terminal.")
args = parser.parse_args()

if __name__ == '__main__':
    if args.file_output:
        with open('output', 'a+') as f:
            with redirect_stdout(f):
                main()
                print('\n')
    else:
        main()
